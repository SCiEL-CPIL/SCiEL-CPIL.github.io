"""
Triangular lattice packing: maximize edges for 32 points.
Combines greedy construction with local search refinement.
Target: 76 edges (Harborth's theorem optimum for n=32).
"""
import time
import random
import math

def _neighbors(p):
    a, b = p
    return ((a + 1, b), (a - 1, b), (a, b + 1),
            (a, b - 1), (a + 1, b - 1), (a - 1, b + 1))

def _dist_sq(p, q):
    """Squared distance between two lattice points a+b*omega."""
    da = p[0] - q[0]
    db = p[1] - q[1]
    # |a + b*omega|^2 = a^2 + ab + b^2 for omega = exp(i*pi/3)
    return da * da + da * db + db * db

def _edge_count(Sset):
    """Count pairs at distance exactly 1."""
    cnt = 0
    # Optimize: iterate over set, check neighbors, divide by 2 or use ordering
    # Since we need speed for many iterations, a direct loop is okay for n=32
    for p in Sset:
        for q in _neighbors(p):
            if q in Sset and p < q:
                cnt += 1
    return cnt

def _valid(Sset):
    """Check validity: 32 distinct points, |a|,|b| <= 64, min distance >= 1."""
    if len(Sset) != 32:
        return False
    if len(set(Sset)) != 32:
        return False
    for a, b in Sset:
        if abs(a) > 64 or abs(b) > 64:
            return False
    pts = list(Sset)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if _dist_sq(pts[i], pts[j]) < 1:
                return False
    return True

def _greedy_construct(seed):
    """
    Greedily construct a set of 32 points.
    At each step, pick the position that maximizes the number of new edges.
    Uses randomization for tie-breaking to explore different shapes.
    """
    rng = random.Random(seed)
    points = set()
    
    # Define a search region for candidates.
    # A hexagon of radius 8 provides plenty of space.
    # Hex distance d = max(|a|, |b|, |a+b|)
    candidates = []
    for a in range(-8, 9):
        for b in range(-8, 9):
            if max(abs(a), abs(b), abs(a+b)) <= 8:
                candidates.append((a, b))
                
    # Pre-compute neighbors for all candidates for speed?
    # Or just compute on the fly. n=32 is small.
    
    for _ in range(32):
        best_score = -1
        best_pos = None
        best_tie_breaker = -1.0
        
        # Iterate through candidates
        # To add randomness, we can shuffle the candidates or just use the tie breaker
        # Shuffling 300+ items 32 times is cheap.
        shuffled_candidates = candidates[:]
        rng.shuffle(shuffled_candidates)
        
        for p in shuffled_candidates:
            if p in points:
                continue
            
            # Check validity: distance >= 1 from all existing points
            # Since lattice points are integers, min dist sq is 1.
            # So we just need p not in points.
            # Wait, the constraint is "mutual distance at least one".
            # On the triangular lattice, distinct integer pairs always have distance >= 1.
            # So distinctness is sufficient.
            
            # Count new edges
            gain = 0
            for nb in _neighbors(p):
                if nb in points:
                    gain += 1
            
            # We want to maximize gain.
            # If gain is higher, it's better.
            # If gain is equal, we want to break ties.
            # A common heuristic for packing is to prefer points that are "central" 
            # or maintain compactness. 
            # However, simple random tie-breaking in the greedy step often works well 
            # with multiple restarts.
            
            # Let's use a slight bias towards points with higher "potential" 
            # i.e., points that have many empty neighbors available for future steps?
            # That's complex. Let's stick to max gain + random tie break.
            
            if gain > best_score:
                best_score = gain
                best_pos = p
                best_tie_breaker = rng.random()
            elif gain == best_score:
                # Tie breaking:
                # 1. Random
                # 2. Prefer points closer to origin? (Compactness)
                # Let's mix: 50% random, 50% compactness
                # Actually, just random is fine if we do many restarts.
                # But to make the greedy step smarter, let's prefer lower hex distance 
                # to keep the shape compact, which helps maximize internal edges.
                hex_dist_p = max(abs(p[0]), abs(p[1]), abs(p[0]+p[1]))
                
                # We don't have the previous best pos's hex dist easily in this loop structure 
                # unless we store it. Let's just use random for now, as the shuffle 
                # already provides some ordering, but we are checking all.
                # Actually, if we shuffle, the first one with max gain is picked.
                # That is effectively random tie-breaking.
                pass # Keep the first one found in shuffled order for max gain
        
        if best_pos is not None:
            points.add(best_pos)
        else:
            # Should not happen with radius 8
            break
            
    return list(points)

def _local_search(Sset, max_iters=5000, deadline=None):
    """
    Fast local search: try moving points to improve edge count.
    Moves: Swap a point with a candidate, or move a point to a neighbor.
    """
    current_edges = _edge_count(Sset)
    best_edges = current_edges
    best_set = list(Sset)
    
    # Candidate positions to try for moves
    # Use a region slightly larger than the current bounding box
    # Or just a fixed reasonable region
    candidates = []
    for a in range(-8, 9):
        for b in range(-8, 9):
            if max(abs(a), abs(b), abs(a+b)) <= 8:
                candidates.append((a, b))
                
    rng = random.Random(999)
    
    # Convert Sset to a list for mutation, keep a set for O(1) lookups
    S_list = list(Sset)
    S_set = set(S_list)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        # Pick a random point to move
        idx = rng.randrange(len(S_list))
        old_pos = S_list[idx]
        
        # Generate move candidates:
        # 1. Neighbors of old_pos (local moves)
        # 2. Random candidates (global moves)
        move_candidates = list(_neighbors(old_pos))
        for _ in range(3):
            move_candidates.append(rng.choice(candidates))
            
        # Shuffle to randomize order
        rng.shuffle(move_candidates)
        
        improved = False
        for new_pos in move_candidates:
            if new_pos == old_pos:
                continue
            if new_pos in S_set:
                continue
            
            # Check validity: distinctness is guaranteed if new_pos not in S_set.
            # Distance constraint is automatically satisfied for distinct lattice points.
            
            # Perform move
            S_list[idx] = new_pos
            S_set.remove(old_pos)
            S_set.add(new_pos)
            
            new_edges = _edge_count(S_set)
            
            if new_edges > current_edges:
                current_edges = new_edges
                improved = True
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_set = list(S_set)
                break # Accept move, restart iteration
            else:
                # Revert
                S_list[idx] = old_pos
                S_set.remove(new_pos)
                S_set.add(old_pos)
                
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    best_result = None
    best_edges = -1
    
    # Strategy: Multiple greedy restarts + local search polish
    # Greedy is fast. Local search is fast for n=32.
    # We can do many iterations.
    
    num_restarts = 200 # Should be fast enough
    
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        seed = i * 12345 + 67890
        initial = _greedy_construct(seed)
        
        if len(initial) != 32:
            continue
            
        # Check edges before search
        edges = _edge_count(initial)
        if edges > best_edges:
            best_edges = edges
            best_result = initial
            
        # If we hit the theoretical max, we can stop
        if best_edges >= 76:
            break
            
        # Local search to polish
        # Use a tighter deadline for each search to allow many restarts
        # Or just let it run until global deadline
        result = _local_search(initial, max_iters=200, deadline=deadline - 5)
        
        edges = _edge_count(result)
        
        if edges > best_edges:
            best_edges = edges
            best_result = result
            
        # If we hit the theoretical max, we can stop
        if best_edges >= 76:
            break
            
    # If we didn't find a good result, try a more exhaustive approach or fallback
    if best_result is None or len(best_result) != 32:
        # Fallback: simple grid or hexagon
        # Hexagon of radius 2 has 19 points.
        # Let's just use a valid grid for safety, though it won't be optimal.
        best_result = [(a, b) for a in range(8) for b in range(4)]
        
    # Final verification
    if not _valid(best_result):
        # This should not happen if our logic is correct
        # But just in case, return a known valid config
        best_result = [(a, b) for a in range(8) for b in range(4)]
        
    return [[a, b] for a, b in best_result]

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Found {len(res)} points")
    print(f"Edges: {_edge_count(res)}")
    print(f"Valid: {_valid(res)}")
