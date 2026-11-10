# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2 using a bitmask-based greedy expansion.
    
    Strategy:
    1. Precompute all 50 directions (49 slopes + vertical) and their line structure.
    2. Start with N = all 2401 points.
    3. Greedily add points to the complement (remove from N) if doing so doesn't
       invalidate any point's last valid line.
    4. Use bitmask representation for speed, enabling many random restarts.
    5. Return the best N found.
    """
    p = 7
    k_const = 3  # alpha^2 = 3
    n_total = 2401
    n_dir = 50  # 49 slopes + vertical
    n_lines_per_dir = 49
    
    # Precompute point indexing
    def point_to_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb
    
    def idx_to_point(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        return (xa, xb, ya, yb)
    
    # Precompute all lines: for each direction d, 49 lines of 49 points each
    # lines[d][l] = array of 49 point indices
    # point_line_id[d][idx] = line_id for point idx in direction d
    
    lines = []
    point_line_id = []
    
    for d in range(n_dir):
        members = []
        lid = np.zeros(n_total, dtype=np.int32)
        
        if d < 49:
            # Direction (1, m) where m = d in 0..48, represented as (ma, mb)
            ma = d % p
            mb = d // p
            
            for c1 in range(p):
                for c2 in range(p):
                    pts = np.empty(49, dtype=np.int32)
                    pt_i = 0
                    for xa in range(p):
                        for xb in range(p):
                            ya = (c1 + xa * ma + xb * mb * k_const) % p
                            yb = (c2 + xa * mb + xb * ma) % p
                            idx = point_to_idx(xa, xb, ya, yb)
                            pts[pt_i] = idx
                            lid[idx] = len(members)
                            pt_i += 1
                    members.append(pts)
        else:
            # Vertical direction (0, 1)
            for xa in range(p):
                for xb in range(p):
                    pts = np.empty(49, dtype=np.int32)
                    pt_i = 0
                    for ya in range(p):
                        for yb in range(p):
                            idx = point_to_idx(xa, xb, ya, yb)
                            pts[pt_i] = idx
                            lid[idx] = len(members)
                            pt_i += 1
                    members.append(pts)
        
        lines.append(members)
        point_line_id.append(lid)
    
    # For each point, precompute which lines it belongs to in each direction
    # point_dir_line[point_idx][d] = line_id
    point_dir_line = np.zeros((n_total, n_dir), dtype=np.int32)
    for d in range(n_dir):
        for idx in range(n_total):
            point_dir_line[idx, d] = point_line_id[d][idx]
    
    # Precompute for each point and direction, the other 48 points on the line
    # other_points[point_idx, d] = array of 48 point indices (excluding point_idx)
    # This is expensive to store (2401 * 50 * 48), so we compute on the fly
    
    def get_line_points(d, line_id):
        return lines[d][line_id]
    
    # Bitmask-based greedy expansion
    # in_N is a boolean array of length n_total
    # clean_mask[point_idx] is a bitmask of directions that are clean for point_idx
    # Direction d is clean for point p if all other 48 points on line(p, d) are in N
    
    def compute_clean_masks(in_N):
        """Compute clean direction bitmask for each point."""
        clean_masks = np.zeros(n_total, dtype=np.int32)
        for d in range(n_dir):
            for l in range(n_lines_per_dir):
                pts = lines[d][l]
                in_N_line = in_N[pts]
                total_in = np.sum(in_N_line)
                if total_in == 49:
                    # All 49 points in N, so direction d is clean for all
                    for pt_idx in pts:
                        clean_masks[pt_idx] |= (1 << d)
                elif total_in == 48:
                    # One point not in N, direction d is clean for all except that point
                    missing = np.where(~in_N_line)[0]
                    if len(missing) == 1:
                        for i in range(49):
                            if i != missing[0]:
                                clean_masks[pts[i]] |= (1 << d)
        return clean_masks
    
    def try_add_to_complement(in_N, clean_masks, point_idx):
        """
        Try to add point_idx to complement (remove from N).
        Returns True if successful, False otherwise.
        Modifies in_N and clean_masks in place if successful.
        """
        # Check if removing point_idx invalidates any point
        # A point q loses direction d if:
        # 1. d was clean for q (before removal)
        # 2. point_idx is on line(q, d)
        # 3. After removal, d is no longer clean for q
        # 4. q had no other clean directions
        
        # For each direction d, check if point_idx is on a line with some q
        # where d was the only clean direction for q
        
        for d in range(n_dir):
            l = point_dir_line[point_idx, d]
            pts = lines[d][l]
            
            # For each point q on this line (except point_idx), check if d was
            # the only clean direction and would become unclean
            for i in range(49):
                qi = pts[i]
                if qi == point_idx:
                    continue
                
                # Check if d was clean for qi before removal
                if not (clean_masks[qi] & (1 << d)):
                    continue
                
                # d was clean for qi. After removing point_idx from N,
                # is d still clean for qi?
                # d is clean for qi iff all other 48 points on line(qi, d) are in N
                # After removal, point_idx is not in N
                # So d is clean for qi after removal iff point_idx is not on line(qi, d)
                # But point_idx IS on line(qi, d) by construction
                # Wait, we need to check: are all OTHER 48 points (excluding qi itself) in N?
                # Before removal: all 48 were in N (since d was clean)
                # After removal: point_idx is removed, so only 47 are in N
                # So d is NO LONGER clean for qi
                
                # So qi loses direction d. Check if qi had any other clean directions
                other_mask = clean_masks[qi] & ~(1 << d)
                if other_mask == 0:
                    # qi had no other clean directions, removal invalid
                    return False
        
        # Removal is valid, apply it
        in_N[point_idx] = False
        clean_masks[point_idx] = 0  # point_idx is no longer in N, no clean directions
        
        # Update clean masks for all other points on lines through point_idx
        for d in range(n_dir):
            l = point_dir_line[point_idx, d]
            pts = lines[d][l]
            
            # For each other point on this line, direction d is no longer clean
            # (since point_idx is removed from N)
            for i in range(49):
                qi = pts[i]
                if qi == point_idx:
                    continue
                clean_masks[qi] &= ~(1 << d)
        
        return True
    
    def validate(in_N):
        """Validate that in_N is a valid Nikodym set."""
        # Every point must have at least one clean direction
        clean_masks = compute_clean_masks(in_N)
        for idx in range(n_total):
            if not (clean_masks[idx] & 0xFFFFFFFF):
                return False
        return True
    
    # Main algorithm: greedy expansion with random restarts
    deadline = 570  # seconds, safety margin before 630s limit
    start_time = time.time()
    
    best_complement_size = 0
    best_in_N = np.ones(n_total, dtype=bool)
    
    # Try multiple random restarts
    rng = np.random.RandomState(42)
    
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() - start_time > deadline - 30:
            break
        
        in_N = np.ones(n_total, dtype=bool)
        clean_masks = compute_clean_masks(in_N)
        
        # Randomly shuffle the order of points to try
        order = np.random.permutation(n_total)
        
        added = 0
        for idx in order:
            if time.time() - start_time > deadline - 30:
                break
            if not in_N[idx]:
                continue
            
            if try_add_to_complement(in_N, clean_masks, idx):
                added += 1
        
        complement_size = np.sum(~in_N)
        if complement_size > best_complement_size:
            best_complement_size = complement_size
            best_in_N = in_N.copy()
        
        # Early termination if we've used most of the budget
        if time.time() - start_time > deadline - 15:
            break
    
    # Build result from best_in_N
    points = []
    for idx in range(n_total):
        if best_in_N[idx]:
            xa, xb, ya, yb = idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])
    
    result = np.array(points, dtype=np.int64)
    return result


# EVOLVE-BLOCK-END
