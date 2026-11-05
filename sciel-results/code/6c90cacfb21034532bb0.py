<<<<<<< SEARCH
    # Sort divisors: prioritize small moduli, with some randomization
    if randomize:
        # Weight by 1/m (smaller moduli get higher weight)
        weights = [1.0 / m for m in divs]
        total_w = sum(weights)
        remaining = list(divs)
        rem_weights = list(weights)
    else:
        remaining = sorted(divs)
    
    while remaining:
        if time.time() > DEADLINE:
            break
        
        if randomize:
            # Adaptive: try top-k candidates, pick best marginal gain
            k = min(10, len(remaining))
            # Sample k candidates weighted by 1/m
            candidates = []
            for _ in range(k):
                r = random.random() * sum(rem_weights)
                cum = 0
                for i, w in enumerate(rem_weights):
                    cum += w
                    if r <= cum:
                        candidates.append(i)
                        break
            # Deduplicate
            candidates = list(set(candidates))
            if not candidates:
                candidates = [0]
        else:
            candidates = list(range(min(50, len(remaining))))
        
        best_gain = -1
        best_idx = -1
        best_res = -1
        
        for idx in candidates:
            m = remaining[idx]
            # Find best residue for this modulus
            best_r = -1
            best_r_gain = -1
            for r in range(m):
                # Count newly covered
                gain = sum(1 for i in range(r, BASE, m) if covered[i] == 0)
                if gain > best_r_gain:
                    best_r_gain = gain
                    best_r = r
            if best_r_gain > best_gain:
                best_gain = best_r_gain
                best_idx = idx
                best_res = best_r
        
        if best_gain <= 0:
            break
        
        m = remaining.pop(best_idx)
        rem_weights.pop(best_idx)
        used_moduli.add(m)
        classes.append([m, best_res])
        covered[best_res::m] = b'\x01' * len(range(best_res, BASE, m))
    
    return classes
=======
    # Sort divisors: prioritize small moduli, with some randomization
    if randomize:
        remaining = list(divs)
        rem_weights = [1.0 / m for m in remaining]
    else:
        remaining = sorted(divs)
        rem_weights = []
    
    # Convert covered to numpy array for faster gain calculation
    covered_np = np.frombuffer(covered, dtype=np.uint8)
    
    while remaining:
        if time.time() > DEADLINE:
            break
        
        if randomize:
            # Adaptive: try top-k candidates, pick best marginal gain
            k = min(10, len(remaining))
            # Sample k candidates weighted by 1/m
            candidates = set()
            total_w = sum(rem_weights)
            if total_w > 0:
                for _ in range(k):
                    r = random.random() * total_w
                    cum = 0
                    for i, w in enumerate(rem_weights):
                        cum += w
                        if r <= cum:
                            candidates.add(i)
                            break
            if not candidates:
                candidates.add(0)
            candidates = list(candidates)
        else:
            candidates = list(range(min(50, len(remaining))))
        
        best_gain = -1
        best_idx = -1
        best_res = -1
        
        for idx in candidates:
            if idx < 0 or idx >= len(remaining):
                continue
            m = remaining[idx]
            # Find best residue for this modulus using numpy
            # We want to maximize the number of 0s covered by this class
            # covered_np[r::m] gives the slice. We want sum(1 - covered_np[r::m])
            # which is len(slice) - sum(covered_np[r::m])
            
            # Optimization: If m is small, iterate residues. If m is large, 
            # the number of residues is small.
            # Actually, for any m, we can just loop r in range(m) and use numpy sum.
            best_r = -1
            best_r_gain = -1
            
            # Pre-calculate total length for this m to avoid repeated len() calls if possible
            # But len(range(r, BASE, m)) varies slightly? No, for fixed m, 
            # len is either floor((BASE-1-r)/m) + 1.
            
            for r in range(m):
                # Get the slice
                # Note: covered_np is 1D. 
                # The indices are r, r+m, r+2m...
                # We can use slicing: covered_np[r::m]
                slice_covered = covered_np[r::m]
                # Gain is number of 0s in this slice
                # gain = len(slice_covered) - np.sum(slice_covered)
                gain = len(slice_covered) - int(np.sum(slice_covered))
                
                if gain > best_r_gain:
                    best_r_gain = gain
                    best_r = r
            
            if best_r_gain > best_gain:
                best_gain = best_r_gain
                best_idx = idx
                best_res = best_r
        
        if best_gain <= 0:
            break
        
        m = remaining.pop(best_idx)
        if randomize:
            rem_weights.pop(best_idx)
        used_moduli.add(m)
        classes.append([m, best_res])
        
        # Update covered array
        # covered[best_res::m] = 1
        # Using numpy for update
        covered_np[best_res::m] = 1
        # Update the bytearray from numpy
        covered = covered_np.tobytes()
        # Re-create numpy view? No, we can just keep covered_np and update it.
        # But the local search and other parts use bytearray.
        # Let's sync them.
        # Actually, _build_coverage returns bytearray. 
        # Let's stick to bytearray for consistency with the rest of the code, 
        # but use numpy for the *calculation* of gain.
        
        # Wait, if I update covered_np, I need to update covered bytearray too if I pass it out.
        # The function returns classes. The caller builds coverage.
        # But inside the loop, I need the updated coverage for the next iteration.
        # So updating covered_np is good.
        # I should also update the bytearray `covered` to keep it in sync if I use it later?
        # I don't use `covered` (bytearray) inside the loop anymore, I use `covered_np`.
        # So I can just return classes.
    
    return classes
>>>>>>> REPLACE

I also need to update `_build_coverage` and `_score` to be consistent or just ensure `construct` handles the conversion. The current `_build_coverage` is fine. The `_local_search` uses bytearray operations which might be slow. Let's optimize `_local_search` similarly if possible, or just ensure the main greedy part is fast enough. The error was the primary blocker.

Let's also fix `_local_search` to use numpy for speed, as it iterates many times.

