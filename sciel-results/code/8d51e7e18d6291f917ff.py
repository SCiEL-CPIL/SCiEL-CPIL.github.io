import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 16
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    # Precompute K4s
    k4s = list(combinations(range(n), 4))
    k4_edge_indices = []
    for quad in k4s:
        e_indices = []
        for i in range(4):
            for j in range(i + 1, 4):
                e_indices.append(edge_index[(quad[i], quad[j])])
        k4_edge_indices.append(e_indices)
        
    num_k4s = len(k4s)
    
    # Map edge to K4s
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_indices in enumerate(k4_edge_indices):
        for e_idx in e_indices:
            edge_to_k4s[e_idx].append(k4_idx)

    def count_colors(colors):
        """Count distinct colors used in the coloring."""
        return len(set(colors))

    def is_valid(colors):
        """Check if every K4 has at least 5 distinct colors."""
        for k4_idx in range(num_k4s):
            e_indices = k4_edge_indices[k4_idx]
            distinct = len(set(colors[e] for e in e_indices))
            if distinct < 5:
                return False
        return True

    def prune_palette(colors, K):
        """
        Try to reduce the number of colors used from K to K-1, K-2, etc.
        We attempt to merge color classes or recolor edges to use fewer colors.
        Returns the pruned colors if successful, else None.
        """
        # Map used colors to 0..K_used-1
        used_colors = sorted(set(colors))
        color_map = {c: i for i, c in enumerate(used_colors)}
        current_K = len(used_colors)
        pruned = [color_map[c] for c in colors]
        
        # Try to reduce K by 1, 2, 3...
        # We use a simple local search to merge colors
        # Objective: minimize number of distinct colors while maintaining validity.
        # This is hard, so we try specific merges.
        
        best_pruned = pruned[:]
        best_K = current_K
        
        # If K is already small, just return.
        if current_K <= 13:
            return best_pruned
            
        # Try to merge the two most frequent colors? Or just try all pairs?
        # Since K is likely 15 or 16 from the algebraic construction, 
        # merging is non-trivial. 
        # Instead, we run a fast local search to minimize palette size.
        
        def run_palette_search(init_colors, max_time=5.0):
            nonlocal best_pruned, best_K
            c = init_colors[:]
            K_curr = count_colors(c)
            if K_curr < best_K:
                best_K = K_curr
                best_pruned = c[:]
            if K_curr == 13:
                return c[:]
                
            # State for fast violation check
            k4_distinct = [0] * num_k4s
            k4_counts = [[0] * 20 for _ in range(num_k4s)] # Max 20 colors
            violating = 0
            
            for k4_idx in range(num_k4s):
                e_indices = k4_edge_indices[k4_idx]
                for ei in e_indices:
                    col = c[ei]
                    if col < 20:
                        k4_counts[k4_idx][col] += 1
                distinct = sum(1 for cnt in k4_counts[k4_idx] if cnt > 0)
                k4_distinct[k4_idx] = distinct
                if distinct < 5:
                    violating += 1
            
            if violating == 0:
                return c[:]

            no_improve = 0
            max_no_improve = 2000
            end_time = time.time() + max_time
            
            while time.time() < end_time:
                if violating == 0:
                    # Try to reduce palette
                    # Find a color with the fewest edges to merge into another?
                    # Or just check if we can recolor an edge to a color already present in its K4s
                    # to reduce global palette.
                    
                    # Simple heuristic: pick a random edge, try to change its color 
                    # to a color that is already used in all its K4s (to not create new distinctness issues)
                    # and ideally a color that is "rare" globally to help pruning?
                    # Actually, to reduce palette, we want to eliminate a color entirely.
                    # Let's identify the color with the smallest frequency.
                    color_freq = [0] * 20
                    for col in c:
                        if col < 20:
                            color_freq[col] += 1
                    
                    # Find least frequent non-zero color
                    min_freq = float('inf')
                    target_color = -1
                    for col in range(20):
                        if color_freq[col] > 0 and color_freq[col] < min_freq:
                            min_freq = color_freq[col]
                            target_color = col
                    
                    if target_color == -1:
                        break
                        
                    # Try to recolor all edges of target_color to other colors
                    edges_to_change = [i for i, col in enumerate(c) if col == target_color]
                    if not edges_to_change:
                        continue
                        
                    changed = False
                    for ei in edges_to_change:
                        # Try to change to a color that is already in the K4s?
                        # Or just random valid change?
                        # To keep it fast, try to map to the most frequent other color
                        # that doesn't break validity.
                        
                        # Simulate change
                        old_col = target_color
                        # Find a candidate new color
                        # Prefer colors that are already in the K4s for this edge
                        candidates = []
                        for k4_idx in edge_to_k4s[ei]:
                            for col in range(20):
                                if col != old_col and k4_counts[k4_idx][col] > 0:
                                    candidates.append(col)
                        
                        if not candidates:
                            # If no candidate, just pick a random color < 20
                            candidates = [random.randint(0, 19) for _ in range(5)]
                        
                        success = False
                        random.shuffle(candidates)
                        for new_col in candidates:
                            if new_col == old_col:
                                continue
                            
                            # Check delta
                            valid_change = True
                            for k4_idx in edge_to_k4s[ei]:
                                # If old_col count becomes 0, distinct decreases by 1
                                # If new_col count was 0, distinct increases by 1
                                # We need distinct >= 5
                                new_dist = k4_distinct[k4_idx]
                                if k4_counts[k4_idx][old_col] == 1:
                                    new_dist -= 1
                                if k4_counts[k4_idx][new_col] == 0:
                                    new_dist += 1
                                if new_dist < 5:
                                    valid_change = False
                                    break
                            
                            if valid_change:
                                # Apply
                                for k4_idx in edge_to_k4s[ei]:
                                    k4_counts[k4_idx][old_col] -= 1
                                    k4_counts[k4_idx][new_col] += 1
                                    new_dist = k4_distinct[k4_idx]
                                    if k4_counts[k4_idx][old_col] == 0:
                                        new_dist -= 1
                                    if k4_counts[k4_idx][new_col] == 1:
                                        new_dist += 1
                                    k4_distinct[k4_idx] = new_dist
                                    if new_dist < 5:
                                        violating += 1
                                    elif k4_distinct[k4_idx] >= 5 and new_dist == 5: # This logic is tricky for tracking violations
                                        pass # We'll re-verify at end of batch or just rely on local check
                                        
                                # Actually, tracking 'violating' incrementally is error-prone.
                                # Let's just apply and if it becomes invalid, revert?
                                # Or simpler: just do the change, and if we end up with violations, 
                                # we accept them temporarily? No, we want to stay valid.
                                
                                # Let's stick to the delta check which ensures validity.
                                c[ei] = new_col
                                changed = True
                                success = True
                                break
                        
                        if not success:
                            # Cannot change this edge safely?
                            # Maybe we should have picked a different target color.
                            # For now, break and try a different target color.
                            break

                    if changed:
                        # Check if target_color is gone
                        if target_color not in c:
                            # Palette reduced!
                            # Remap colors to 0..K-1
                            used = sorted(set(c))
                            cmap = {col: i for i, col in enumerate(used)}
                            c = [cmap[col] for col in c]
                            K_curr = len(used)
                            if K_curr < best_K:
                                best_K = K_curr
                                best_pruned = c[:]
                                if K_curr == 13:
                                    return c[:]
                            # Re-initialize state arrays for next iteration?
                            # Too slow. Just continue.
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
                        
                    if no_improve > max_no_improve:
                        break
                        
            return c[:] if is_valid(c) else None

        # Start palette search from the algebraic coloring
        pruned_res = run_palette_search(pruned)
        if pruned_res:
            return pruned_res
            
        return best_pruned

    # --- Algebraic Construction ---
    # Vertices are elements of GF(4)^2.
    # GF(4) = {0, 1, w, w^2} with w^2 + w + 1 = 0.
    # Elements: 0, 1, 2 (w), 3 (w^2)
    # Multiplication table for GF(4):
    # 0*x = 0
    # 1*x = x
    # 2*2 = 3 (w*w = w^2)
    # 2*3 = 1 (w*w^2 = w^3 = 1)
    # 3*3 = 2 (w^2*w^2 = w^4 = w)
    
    def gf4_add(a, b):
        # XOR in GF(2)^2 representation
        # 0=00, 1=01, 2=10, 3=11
        return a ^ b

    def gf4_mul(a, b):
        # 0 -> 0
        if a == 0 or b == 0:
            return 0
        # 1 -> identity
        if a == 1:
            return b
        if b == 1:
            return a
        # 2 (w) * 2 (w) = 3 (w^2)
        # 2 (w) * 3 (w^2) = 1
        # 3 (w^2) * 3 (w^2) = 2 (w)
        if a == 2 and b == 2:
            return 3
        if a == 2 and b == 3:
            return 1
        if a == 3 and b == 2:
            return 1
        if a == 3 and b == 3:
            return 2
        return 0

    # Vertices 0..15 map to (x, y) in GF(4)^2
    # v = 4*x + y
    # Edge color: (x1 - x2) * (y1 - y2) ? No, that's 0 if x1=x2 or y1=y2.
    # We need a non-degenerate bilinear form or similar.
    # Consider color = (x1*x2 + y1*y2) ? No.
    
    # Let's use the property of affine planes.
    # In AG(2, 4), lines have 4 points.
    # A K4 is a set of 4 points.
    # If we color edge (u, v) by the "direction" of the line uv, 
    # there are 5 directions (slopes) in AG(2, 4).
    # Slopes in GF(4): 0, 1, w, w^2, infinity.
    # This gives exactly 5 colors!
    # But we need EVERY K4 to see 5 colors.
    # If a K4 is a line, all edges have the same color (1 color). FAIL.
    # So we must perturb the coloring.
    
    # Strategy: Use the 5 slope colors as a base, but add a secondary component
    # to ensure that any 4 points (even a line) get distinct colors.
    
    # Let's try: Color(u, v) = Slope(u, v) * 3 + (u_id + v_id) % 3 ?
    # Total colors = 5 * 3 = 15.
    # If u, v, w, z are collinear, Slopes are all S.
    # Colors are 3S + (u+v)%3, 3S + (u+w)%3, etc.
    # If (u+v)%3, (u+w)%3, (u+z)%3, (v+w)%3, (v+z)%3, (w+z)%3 are all distinct mod 3?
    # No, only 3 residues. So we'd have at most 3 distinct colors from the second part.
    # Plus the slope part is constant. So 3 colors. FAIL.
    
    # We need more variation.
    # Color(u, v) = 5 * ((u_id + v_id) % 3) + Slope(u, v)
    # Range 0..14.
    # For collinear points, Slope is constant S.
    # Colors: 5*(u+v)%3 + S, 5*(u+w)%3 + S, ...
    # The terms (u+v)%3 take values in {0,1,2}.
    # So we get at most 3 distinct colors: 5*0+S, 5*1+S, 5*2+S.
    # Still fails.
    
    # We need the secondary part to distinguish ALL 6 edges of a K4.
    # The secondary part must be injective on the set of 6 edges of any K4?
    # No, just distinct enough to reach 5.
    
    # Let's use: Color(u, v) = 15 * Slope(u, v) + (u_id * 3 + v_id) % 15 ?
    # Too many colors.
    
    # Let's try a different algebraic structure.
    # Use GF(16) ? No, vertices are 16.
    # Map vertices to GF(16).
    # Edge color = u + v ?
    # If u, v, w, z are in GF(16).
    # Colors: u+v, u+w, u+z, v+w, v+z, w+z.
    # Are these distinct?
    # u+v = u+w => v=w.
    # u+v = u+z => v=z.
    # u+v = v+w => u=w.
    # u+v = v+z => u=z.
    # u+v = w+z => u+v = w+z.
    # In GF(16), if {u,v,w,z} are distinct, u+v = w+z is possible (harmonic quadruples).
    # If u+v = w+z, we have a collision.
    # How many collisions?
    # If we have a collision, we lose 1 distinct color. Max 5 distinct.
    # We need AT LEAST 5. So 1 collision is OK (5 distinct). 2 collisions is BAD (4 distinct).
    # Can we have 2 collisions in a K4?
    # u+v = w+z AND u+w = v+z => v=w (impossible).
    # u+v = w+z AND u+z = v+w => v=w (impossible).
    # u+v = w+z AND v+w = u+z => u=v (impossible).
    # So at most 1 pair of equal sums in a set of 4 distinct elements in a field?
    # Actually, if u+v = w+z, then u-w = z-v.
    # Can we have u+v = w+z AND u+w = v+z? No.
    # Can we have u+v = w+z AND v+z = u+w? No.
    # What about u+v = w+z AND u+z = v+w?
    # u+v = w+z => u-w = z-v
    # u+z = v+w => u-v = w-z
    # Adding: 2u - w - v = 0 => 2u = v+w.
    # In char 2, 2u=0, so v+w=0 => w=v. Impossible.
    # So in char 2, no two sums can be equal?
    # Wait, u+v = w+z implies u+w = v+z?
    # u+v = w+z => u+w+v+z = 0.
    # u+w = v+z => u+w+v+z = 0.
    # Yes! In characteristic 2, u+v = w+z IF AND ONLY IF u+w = v+z IF AND ONLY IF u+z = v+w.
    # So if one sum equality holds, ALL three hold.
    # This means the 6 sums collapse into 3 pairs of equal values.
    # Distinct colors = 3. FAIL.
    
    # So u+v in GF(16) is bad.
    
    # Try u*v in GF(16)?
    # 0 is a problem.
    # Use u+v+1 ?
    # Same issue.
    
    # Let's go back to AG(2, 4) with 15 colors.
    # Color(u, v) = 3 * Slope(u, v) + (u_id + v_id) % 3
    # We established this gives 3 colors for lines.
    # What if we use (u_id * 2 + v_id) % 3 ?
    # Still 3 residues.
    
    # We need a function f(u, v) such that for any 4 collinear points, 
    # the values f(u,v) are all distinct?
    # There are 6 edges. We need 5 distinct.
    # If we use 3 colors for slope, and 5 colors for "position", total 15.
    # Color = 5 * Slope + Pos(u, v) % 5.
    # For a line, Slope is constant. Pos(u, v) must take at least 5 distinct values mod 5.
    # Pos(u, v) = (u_id + 2*v_id) % 5 ?
    # For 4 collinear points p1, p2, p3, p4 in GF(4)^2.
    # They are of the form p0 + t*d for t in GF(4).
    # u_id, v_id are just integers 0..15.
    # This integer mapping doesn't respect the field structure.
    
    # Let's map vertices to GF(4)^2 properly.
    # v = (x, y).
    # Line: x = ax + b ? Or y = mx + c.
    # Let's use the field coordinates directly.
    # Pos(u, v) = (x_u + y_v) in GF(4)? No, need 5 values.
    # GF(4) has 4 elements.
    # We need 5 distinct values for the 6 edges of a line.
    # This is impossible if Pos maps to GF(4) (4 values).
    # We need Pos to map to a set of size >= 5.
    # Use integers 0..4.
    # Pos(u, v) = (x_u * 4 + y_u + x_v * 2 + y_v) % 5 ?
    # Arbitrary.
    
    # Alternative:
    # Use 15 colors directly.
    # Color(u, v) = (x_u - x_v) * 3 + (y_u - y_v) * 1 ?
    # In GF(4), differences are in GF(4).
    # (x_u - x_v) has 4 values. (y_u - y_v) has 4 values.
    # Pairs (dx, dy) have 16 values. One is (0,0) which is u=v.
    # So 15 non-zero vectors.
    # Color = index of vector (dx, dy) in GF(4)^2 \ {(0,0)}.
    # This is a 15-coloring.
    # For any K4, do we see 5 colors?
    # If u, v, w, z are collinear, then (x_u-x_v, y_u-y_v) are all scalar multiples of (x_w-x_v, y_w-y_v).
    # In GF(4), non-zero vectors on a line through origin are {d, 2d, 3d} (since 1, 2, 3 are non-zero scalars).
    # Wait, GF(4) has 3 non-zero elements: 1, 2, 3.
    # So the differences between 4 collinear points will be among {d, 2d, 3d, -d, -2d, -3d}.
    # In char 2, -d = d.
    # So the differences are {d, 2d, 3d}.
    # There are only 3 distinct direction vectors!
    # So a line of 4 points sees only 3 colors. FAIL.
    
    # We need to break the symmetry.
    # Color(u, v) = index( (x_u - x_v, y_u - y_v) ) + (u_id + v_id) % 3 ?
    # Total colors = 15 * 3 = 45. Too many.
    # We want ~13-15.
    
    # Let's try: Color(u, v) = index( (x_u - x_v, y_u - y_v) ) % 5 ?
    # If we map the 15 non-zero vectors to 5 colors (3 per color),
    # a line sees 3 vectors, which might map to 1, 2, or 3 colors.
    # If they map to 1 color, FAIL.
    
    # What if we use: Color(u, v) = index( (x_u - x_v, y_u - y_v) ) // 3 ?
    # 15 vectors / 3 = 5 colors.
    # If a line's 3 vectors fall in the same bucket, FAIL.
    
    # We need a function f: GF(4)^2 \ {0} -> {0..K-1} such that
    # for any line L, the set { f(v1-v2) for v1,v2 in L } has size >= 5?
    # No, L has 4 points, so 6 differences, but only 3 distinct vectors.
    # So max distinct colors from a line is 3.
    # So ANY coloring based solely on difference (translation invariant) fails for lines.
    # We MUST use vertex IDs or position.
    
    # Color(u, v) = 3 * index(diff) + (u_id % 3) ?
    # For a line, diff index takes 3 values.
    # u_id % 3 takes 3 values.
    # Total 9 combos.
    # If u_id % 3 is constant for the 4 points?
    # If we pick 4 points with same u_id % 3, then Color = 3 * idx.
    # idx takes 3 values. So 3 colors. FAIL.
    
    # Color(u, v) = 5 * (u_id % 3) + index(diff) % 5 ?
    # For a line, index(diff) % 5 takes 3 values (since 3 distinct vectors).
    # u_id % 3 takes 3 values.
    # If we pick points such that u_id % 3 is constant, we get 3 colors.
    # If we pick points such that u_id % 3 varies, we get more.
    # But an adversary (or random K4) can pick 4 points with same u_id % 3.
    # There are 16 points. 16/3 = 5.3. So there are at least 5 points with u_id % 3 = 0.
    # Any 4 of them form a K4.
    # If those 4 are collinear, they fail.
    # Are there 4 collinear points with same u_id % 3?
    # Likely yes.
    
    # We need a more robust perturbation.
    # Color(u, v) = (u_id + v_id) % 13 + index(diff) % 13 ?
    # 13 colors.
    # For a line, index(diff) is constant (3 values).
    # (u_id + v_id) % 13 varies.
    # If we pick 4 collinear points, do the sums (u+v)%13 give 5 distinct values?
    # u, v, w, z collinear.
    # Sums: u+v, u+w, u+z, v+w, v+z, w+z.
    # In GF(4) coordinates, u = p0 + t1*d, v = p0 + t2*d, etc.
    # u+v = 2p0 + (t1+t2)d = (t1+t2)d (in char 2).
    # So sums are (t1+t2)d, (t1+t3)d, etc.
    # t1, t2, t3, t4 are distinct in GF(4).
    # Sums t1+t2, t1+t3, t1+t4, t2+t3, t2+t4, t3+t4.
    # In GF(4), t1+t2 = t3+t4?
    # t1+t2+t3+t4 = 0+1+2+3 = 0 (in GF(4), 0+1+2+3 = 0+1+w+w^2 = 0).
    # So t1+t2 = t3+t4.
    # So we have 3 pairs of equal sums.
    # So (u+v)%13 takes 3 distinct values.
    # index(diff) takes 3 distinct values.
    # Total colors = 3 + 3 = 6? No, they are added.
    # If A takes 3 values and B takes 3 values, A+B takes at most 6 values.
    # We need 5.
    # Is it guaranteed to be 5?
    # If A = {0, 1, 2} and B = {0, 1, 2}, A+B = {0, 1, 2, 3, 4}. Size 5.
    # If A = {0, 1, 2} and B = {0, 2, 4}, A+B = {0, 1, 2, 3, 4, 6}. Size 6.
    # So it seems likely to be >= 5.
    
    # Let's try this:
    # Map vertices to GF(4)^2.
    # Color(u, v) = ( (u_id + v_id) % 13 ) + ( index(diff(u,v)) % 13 )
    # Mod 13?
    # If we don't mod, range is 0..24.
    # We want to minimize colors.
    # If we use mod 13, we might collide.
    # Let's just use the sum without mod, and see how many colors we use.
    # Max 24.
    # Then prune.
    
    # Implementation:
    # 1. Map 0..15 to GF(4)^2.
    # 2. Compute diff index for each edge.
    # 3. Color = (u_id + v_id) + 5 * diff_index.
    # 4. Prune.
    
    # Let's define the mapping.
    # v = 4*x + y.
    # x, y in {0,1,2,3}.
    # diff = (x1-x2, y1-y2).
    # Non-zero diffs: 15.
    # Index them 0..14.
    
    diff_to_idx = {}
    idx = 0
    for dx in range(4):
        for dy in range(4):
            if dx == 0 and dy == 0:
                continue
            diff_to_idx[(dx, dy)] = idx
            idx += 1
            
    colors = [0] * num_edges
    for e_idx, (u, v) in enumerate(edges):
        x_u, y_u = divmod(u, 4)
        x_v, y_v = divmod(v, 4)
        dx = x_u ^ x_v # GF(4) addition is XOR
        dy = y_u ^ y_v
        d_idx = diff_to_idx.get((dx, dy), 0)
        # Color = (u + v) + 5 * d_idx
        colors[e_idx] = (u + v) + 5 * d_idx
        
    # Verify
    if is_valid(colors):
        pruned = prune_palette(colors, count_colors(colors))
        if pruned:
            return pruned
        return colors
        
    # If invalid, fallback to random search with K=15
    # (Reuse logic from seed if needed, but let's try a simple random init first)
    
    def run_search(K, time_limit, init_seed=None):
        if init_seed is not None:
            random.seed(init_seed)
        colors = [random.randint(0, K-1) for _ in range(num_edges)]
        
        k4_counts = [[0] * K for _ in range(num_k4s)]
        k4_distinct = [0] * num_k4s
        violating_k4s = set()
        cost = 0
        
        for k4_idx in range(num_k4s):
            e_indices = k4_edge_indices[k4_idx]
            for ei in e_indices:
                c = colors[ei]
                k4_counts[k4_idx][c] += 1
            distinct = sum(1 for c in range(K) if k4_counts[k4_idx][c] > 0)
            k4_distinct[k4_idx] = distinct
            if distinct < 5:
                violating_k4s.add(k4_idx)
                cost += 1
                
        if cost == 0:
            return colors[:]
            
        best_cost = cost
        best_colors = colors[:]
        no_improve_count = 0
        max_no_improve = 5000
        
        def compute_delta(e_idx, old_color, new_color):
            delta = 0
            for k4_idx in edge_to_k4s[e_idx]:
                old_dist = k4_distinct[k4_idx]
                old_count = k4_counts[k4_idx][old_color]
                new_count = k4_counts[k4_idx][new_color]
                
                new_dist = old_dist
                if old_count == 1:
                    new_dist -= 1
                if new_count == 0:
                    new_dist += 1
                    
                if old_dist < 5 and new_dist >= 5:
                    delta -= 1
                elif old_dist >= 5 and new_dist < 5:
                    delta += 1
            return delta

        def apply_change(e_idx, new_color):
            nonlocal cost
            old_color = colors[e_idx]
            if old_color == new_color:
                return 0
                
            delta = 0
            for k4_idx in edge_to_k4s[e_idx]:
                old_dist = k4_distinct[k4_idx]
                
                k4_counts[k4_idx][old_color] -= 1
                k4_counts[k4_idx][new_color] += 1
                
                new_dist = old_dist
                if k4_counts[k4_idx][old_color] == 0 and old_dist > 0:
                    new_dist -= 1
                if k4_counts[k4_idx][new_color] == 1:
                    new_dist += 1
                    
                k4_distinct[k4_idx] = new_dist
                
                if old_dist < 5 and new_dist >= 5:
                    delta -= 1
                elif old_dist >= 5 and new_dist < 5:
                    delta += 1
                    
                if old_dist < 5 and new_dist >= 5:
                    violating_k4s.discard(k4_idx)
                elif old_dist >= 5 and new_dist < 5:
                    violating_k4s.add(k4_idx)
                    
            colors[e_idx] = new_color
            cost += delta
            return delta

        while time.time() < time_limit:
            if violating_k4s and random.random() < 0.7:
                k4_idx = random.choice(list(violating_k4s))
                e_indices = k4_edge_indices[k4_idx]
                e_idx = random.choice(e_indices)
            else:
                e_idx = random.randint(0, num_edges - 1)
                
            new_color = random.randint(0, K - 1)
            if new_color == colors[e_idx]:
                continue
                
            old_color = colors[e_idx]
            delta = compute_delta(e_idx, old_color, new_color)
            
            if delta <= 0:
                apply_change(e_idx, new_color)
                
                if cost < best_cost:
                    best_cost = cost
                    best_colors = colors[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                if cost == 0:
                    return colors[:]
            else:
                if random.random() < 0.05:
                    apply_change(e_idx, new_color)
                    no_improve_count += 1
                else:
                    no_improve_count += 1
                    
            if no_improve_count > max_no_improve:
                colors = [random.randint(0, K-1) for _ in range(num_edges)]
                for k4_idx in range(num_k4s):
                    e_indices = k4_edge_indices[k4_idx]
                    for ei in e_indices:
                        c = colors[ei]
                        k4_counts[k4_idx][c] += 1
                    distinct = sum(1 for c in range(K) if k4_counts[k4_idx][c] > 0)
                    k4_distinct[k4_idx] = distinct
                    if distinct < 5:
                        violating_k4s.add(k4_idx)
                    else:
                        violating_k4s.discard(k4_idx)
                cost = len(violating_k4s)
                best_cost = cost
                best_colors = colors[:]
                no_improve_count = 0
                if cost == 0:
                    return colors[:]
                    
        return best_colors if best_cost == 0 else None

    # Fallback search
    remaining = deadline - time.time() - 5
    if remaining > 5:
        res = run_search(15, remaining, init_seed=42)
        if res:
            return res
            
    # Last resort
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
