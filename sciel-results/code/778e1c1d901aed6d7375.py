# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 570  # safety margin
    
    # Generate all vectors with exactly 2 non-zero entries (±1)
    # There are C(11,2) * 4 = 55 * 4 = 220 such vectors
    def generate_vectors_2():
        vecs = []
        for combo in itertools.combinations(range(d), 2):
            for signs in itertools.product([1, -1], repeat=2):
                v = np.zeros(d, dtype=np.int64)
                v[combo[0]] = signs[0]
                v[combo[1]] = signs[1]
                vecs.append(v)
        return vecs
    
    def generate_vectors_4():
        vecs = []
        for combo in itertools.combinations(range(d), 4):
            for signs in itertools.product([1, -1], repeat=4):
                v = np.zeros(d, dtype=np.int64)
                for idx, val in zip(combo, signs):
                    v[idx] = val
                vecs.append(v)
        return vecs
    
    def generate_vectors_6():
        vecs = []
        for combo in itertools.combinations(range(d), 6):
            for signs in itertools.product([1, -1], repeat=6):
                v = np.zeros(d, dtype=np.int64)
                for idx, val in zip(combo, signs):
                    v[idx] = val
                vecs.append(v)
        return vecs
    
    # Pre-generate candidate sets
    cands_2 = generate_vectors_2()   # 220 vectors, norm² = 2
    cands_4 = generate_vectors_4()   # C(11,4)*16 = 330*16 = 5280 vectors, norm² = 4
    cands_6 = generate_vectors_6()   # C(11,6)*64 = 462*64 = 29568 vectors, norm² = 6
    
    # For the constraint: min squared distance >= max squared norm
    # If we use vectors of norm² = 4, then max_norm = 4
    # We need ||u-v||² >= 4 for all pairs
    # ||u-v||² = ||u||² + ||v||² - 2*u·v = 4 + 4 - 2*u·v = 8 - 2*u·v
    # So we need 8 - 2*u·v >= 4 => u·v <= 2
    
    # If we use vectors of norm² = 2, then max_norm = 2
    # We need ||u-v||² >= 2
    # ||u-v||² = 2 + 2 - 2*u·v = 4 - 2*u·v
    # So we need 4 - 2*u·v >= 2 => u·v <= 1
    
    # Strategy: Use bitset-accelerated greedy to find large independent sets
    # For norm²=4 vectors with dot <= 2 constraint
    
    def greedy_bitset(candidates, max_dot, max_norm):
        """
        Greedy selection using bitset acceleration.
        candidates: list of np arrays
        max_dot: maximum allowed dot product between any two selected vectors
        max_norm: squared norm of each candidate (assumed uniform)
        """
        n_cands = len(candidates)
        if n_cands == 0:
            return []
        
        # Precompute dot products matrix (may be large, so do it in chunks)
        # Actually, for 5280 vectors, that's 5280*5280/2 ≈ 14M pairs, which is manageable
        # But storing the full matrix is 5280*5280 = 28M entries, ~224MB for int8
        # Let's use a different approach: maintain a set of forbidden candidates
        
        selected = []
        selected_indices = []
        
        # For efficiency, we'll track which candidates are still available
        available = [True] * n_cands
        
        for i in range(n_cands):
            if time.time() > deadline - 30:
                break
            if not available[i]:
                continue
            
            v = candidates[i]
            selected.append(v)
            selected_indices.append(i)
            
            # Update availability of all subsequent candidates
            for j in range(i + 1, n_cands):
                if available[j]:
                    dot = int(np.dot(v, candidates[j]))
                    if dot > max_dot:
                        available[j] = False
        
        return selected
    
    def greedy_bitset_fast(candidates, max_dot):
        """
        Faster greedy using precomputed dot product structure.
        """
        n_cands = len(candidates)
        if n_cands == 0:
            return []
        
        # Convert to numpy for faster dot products
        cand_arr = np.array(candidates, dtype=np.int64)
        
        selected = []
        available_mask = np.ones(n_cands, dtype=bool)
        
        for i in range(n_cands):
            if time.time() > deadline - 30:
                break
            if not available_mask[i]:
                continue
            
            v = cand_arr[i]
            selected.append(v)
            
            # Compute dots with all remaining available candidates
            remaining_idx = np.where(available_mask)[0]
            remaining_idx = remaining_idx[remaining_idx > i]
            
            if len(remaining_idx) > 0:
                dots = cand_arr[remaining_idx] @ v
                too_close = dots > max_dot
                if np.any(too_close):
                    available_mask[remaining_idx[too_close]] = False
        
        return selected
    
    # Try multiple strategies
    
    best_points = []
    best_score = 0
    
    # Strategy 1: Greedy on norm²=4 vectors with dot <= 2
    # This should give a good independent set
    print("Strategy 1: Greedy on norm²=4 vectors with dot<=2")
    result4 = greedy_bitset_fast(cands_4, max_dot=2)
    if len(result4) > best_score:
        best_points = list(result4)
        best_score = len(result4)
    print(f"  Found {best_score} points")
    
    # Strategy 2: Try to add norm²=2 vectors to the norm²=4 set
    # If we have max_norm = 4 (from norm²=4 vectors), then for a norm²=2 vector w:
    # ||u - w||² = ||u||² + ||w||² - 2*u·w = 4 + 2 - 2*u·w = 6 - 2*u·w
    # Need 6 - 2*u·w >= 4 => u·w <= 1
    # And for two norm²=2 vectors w1, w2:
    # ||w1 - w2||² = 2 + 2 - 2*w1·w2 = 4 - 2*w1·w2
    # Need 4 - 2*w1·w2 >= 4 => w1·w2 <= 0
    
    print("Strategy 2: Add norm²=2 vectors to norm²=4 set")
    if best_points:
        existing = np.array(best_points, dtype=np.int64)
        existing_norms = np.sum(existing * existing, axis=1)
        max_norm = int(np.max(existing_norms))
        
        new_points = list(best_points)
        
        for w in cands_2:
            if time.time() > deadline - 30:
                break
            valid = True
            # Check against existing points
            dots = existing @ w
            for i, dot in enumerate(dots):
                need = (existing_norms[i] + 2 - max_norm) / 2.0
                if dot > need:
                    valid = False
                    break
            if valid:
                # Check against newly added norm²=2 points
                norm2_added = [p for p in new_points[len(best_points):]]
                if norm2_added:
                    for p in norm2_added:
                        dot2 = int(np.dot(w, p))
                        if dot2 > 0:  # need dot <= 0 for norm²=2 pairs
                            valid = False
                            break
            if valid:
                new_points.append(w)
        
        if len(new_points) > best_score:
            best_points = new_points
            best_score = len(new_points)
    print(f"  Best so far: {best_score} points")
    
    # Strategy 3: Pure norm²=2 greedy with dot <= 1
    print("Strategy 3: Pure norm²=2 greedy with dot<=1")
    result2 = greedy_bitset_fast(cands_2, max_dot=1)
    if len(result2) > best_score:
        best_points = list(result2)
        best_score = len(result2)
    print(f"  Best so far: {best_score} points")
    
    # Strategy 4: Try norm²=6 vectors
    # For norm²=6, need ||u-v||² >= 6
    # ||u-v||² = 6+6-2*u·v = 12-2*u·v >= 6 => u·v <= 3
    print("Strategy 4: Greedy on norm²=6 vectors with dot<=3")
    if time.time() < deadline - 60:
        result6 = greedy_bitset_fast(cands_6, max_dot=3)
        if len(result6) > best_score:
            best_points = list(result6)
            best_score = len(result6)
    print(f"  Best so far: {best_score} points")
    
    # Strategy 5: Mixed - start with norm²=4, then try adding norm²=6
    # If max_norm = 6, then for norm²=4 vector u and norm²=6 vector v:
    # ||u-v||² = 4+6-2*u·v = 10-2*u·v >= 6 => u·v <= 2
    # For two norm²=6 vectors: 12-2*u·v >= 6 => u·v <= 3
    # For two norm²=4 vectors: 8-2*u·v >= 6 => u·v <= 1
    # This is complex, let's try a simpler mixed approach
    
    # Strategy 6: Local search / hill climbing on the norm²=4 solution
    # Try to swap out one vector for two others
    if best_score > 0 and time.time() < deadline - 30:
        print("Strategy 6: Local search on current best")
        current = np.array(best_points, dtype=np.int64)
        n = len(current)
        norms = np.sum(current * current, axis=1)
        max_norm = int(np.max(norms))
        
        # Build a set of selected vectors for fast lookup
        selected_set = set()
        for i in range(n):
            selected_set.add(tuple(current[i].tolist()))
        
        improved = True
        while improved and time.time() < deadline - 30:
            improved = False
            # Try removing one vector and adding two others
            for remove_idx in range(n):
                if time.time() > deadline - 30:
                    break
                v_remove = current[remove_idx]
                
                # Find candidates that are compatible with all other selected vectors
                # but not currently selected
                compatible_cands = []
                for cand in cands_4:
                    if tuple(cand.tolist()) in selected_set:
                        continue
                    # Check compatibility with all vectors except the one being removed
                    valid = True
                    for j in range(n):
                        if j == remove_idx:
                            continue
                        dot = int(np.dot(cand, current[j]))
                        need = (4 + 4 - max_norm) / 2.0
                        if dot > need:
                            valid = False
                            break
                    if valid:
                        compatible_cands.append(cand)
                
                # Try to find two compatible candidates that are also compatible with each other
                for a_idx in range(len(compatible_cands)):
                    if time.time() > deadline - 30:
                        break
                    for b_idx in range(a_idx + 1, len(compatible_cands)):
                        dot_ab = int(np.dot(compatible_cands[a_idx], compatible_cands[b_idx]))
                        if dot_ab <= 2:
                            # Found a valid swap
                            new_points = list(current)
                            del new_points[remove_idx]
                            new_points.append(compatible_cands[a_idx])
                            new_points.append(compatible_cands[b_idx])
                            current = np.array(new_points, dtype=np.int64)
                            n = len(current)
                            norms = np.sum(current * current, axis=1)
                            max_norm = int(np.max(norms))
                            selected_set = set()
                            for i in range(n):
                                selected_set.add(tuple(current[i].tolist()))
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if n > best_score:
            best_points = list(current)
            best_score = n
    print(f"  Best so far: {best_score} points")
    
    # Final validation
    if best_points:
        arr = np.array(best_points, dtype=np.int64)
        n = len(arr)
        norms = np.sum(arr * arr, axis=1)
        max_norm = int(np.max(norms))
        
        # Verify all pairs
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                diff = arr[i] - arr[j]
                sq_dist = int(np.sum(diff * diff))
                if sq_dist < max_norm:
                    valid = False
                    break
            if not valid:
                break
        
        if valid and n >= 2:
            result = np.around(arr).astype(np.int64)
            return result
    
    # Fallback: return the 220 norm²=2 vectors
    return np.array(cands_2, dtype=np.int64)

# EVOLVE-BLOCK-END
