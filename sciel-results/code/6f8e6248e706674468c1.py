import numpy as np
import time
from itertools import combinations, product
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safe margin before 150s kill

    def verify(n, colors):
        """Verify coloring has no red K4 and no blue K5."""
        # Precompute triple indices for speed
        triple_to_idx = {}
        idx = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    triple_to_idx[(a, b, c)] = idx
                    idx += 1
        
        # Check Red K4
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if (colors[triple_to_idx[(a, b, c)]] == 0 and
                colors[triple_to_idx[(a, b, d)]] == 0 and
                colors[triple_to_idx[(a, c, d)]] == 0 and
                colors[triple_to_idx[(b, c, d)]] == 0):
                return False

        # Check Blue K5
        for quint in combinations(range(n), 5):
            all_blue = True
            for triple in combinations(quint, 3):
                if colors[triple_to_idx[triple]] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False

        return True

    def try_algebraic(n, time_limit):
        """
        Search for colorings of the form:
        color(a,b,c) = 0 if (L1*a + L2*b + L3*c) % M is in RedSet else 1
        
        We iterate over M in [2, 3, 4, 5, 6, 7, 8] and subsets of residues for Red.
        """
        # Optimize verification by precomputing violations for a given coloring generator
        # But since we are generating from small parameters, we can just generate and verify.
        # To speed up, we can check violations incrementally or just rely on verify() being fast enough for n=34?
        # verify() for n=34:
        # K4s: C(34,4) = 46376. Each check 4 lookups. ~200k ops.
        # K5s: C(34,5) = 278256. Each check 10 lookups. ~2.8M ops.
        # Total ~3M ops per verification. In Python, this might take ~0.1s - 0.5s.
        # Number of candidates:
        # M=3: 3^3 coeffs * 2^3 subsets = 27 * 8 = 216.
        # M=4: 4^3 coeffs * 2^4 subsets = 64 * 16 = 1024.
        # M=5: 5^3 coeffs * 2^5 subsets = 125 * 32 = 4000.
        # Total candidates ~5000. 5000 * 0.1s = 500s. Too slow if verify is that slow.
        
        # Optimization: Use numpy for vectorized verification?
        # Or early exit in verify?
        # Let's implement a faster verify using bitsets or just optimized loops.
        
        # Even better: Check K4 and K5 constraints during generation? No, generation is cheap.
        # We need a faster verify.
        
        def fast_verify(n, colors):
            # colors is a list of 0s and 1s
            # Convert to numpy array for faster indexing? 
            # Actually, pure python loops are okay if optimized.
            
            # Build index map
            t_to_i = {}
            idx = 0
            for a in range(n):
                for b in range(a+1, n):
                    for c in range(b+1, n):
                        t_to_i[(a, b, c)] = idx
                        idx += 1
            # t_to_i size C(n,3). For n=34, 5984. Dict lookup is fast.
            
            # Check Red K4
            for quad in combinations(range(n), 4):
                a, b, c, d = quad
                # Accessing dict is O(1)
                if (colors[t_to_i[(a, b, c)]] == 0 and
                    colors[t_to_i[(a, b, d)]] == 0 and
                    colors[t_to_i[(a, c, d)]] == 0 and
                    colors[t_to_i[(b, c, d)]] == 0):
                    return False

            # Check Blue K5
            for quint in combinations(range(n), 5):
                # Get all 10 triple indices
                q0, q1, q2, q3, q4 = quint
                idxs = [
                    t_to_i[(q0, q1, q2)], t_to_i[(q0, q1, q3)], t_to_i[(q0, q1, q4)],
                    t_to_i[(q0, q2, q3)], t_to_i[(q0, q2, q4)], t_to_i[(q0, q3, q4)],
                    t_to_i[(q1, q2, q3)], t_to_i[(q1, q2, q4)], t_to_i[(q1, q3, q4)],
                    t_to_i[(q2, q3, q4)]
                ]
                # Check if all are 1
                # Using sum is faster than loop with break in some cases, or just any
                if all(colors[i] == 1 for i in idxs):
                    return False

            return True

        # Search parameters
        # We can prune candidates early if we find a K4 or K5 in a small subset?
        # Let's just iterate.
        
        # To make it faster, we can try to generate the coloring and check violations
        # on a random subset of K4/K5s first?
        # No, we need certainty.
        
        # Let's try to optimize the inner loop of generation.
        # For a fixed (M, L1, L2, L3, RedSet), generate colors.
        # Instead of storing full colors and verifying, we can check constraints on the fly?
        # No, constraints are global.
        
        # Strategy:
        # 1. Iterate M from 2 to 8.
        # 2. Iterate L1, L2, L3 in range(M).
        # 3. Iterate RedSet (subsets of 0..M-1).
        #    We can skip symmetric cases (RedSet vs BlueSet swap) by fixing one property,
        #    e.g., 0 is in RedSet or not? Or just check all.
        # 4. Generate colors array.
        # 5. Fast verify.
        
        # To speed up verify, we can pre-compute the triple indices list once for n.
        triples = list(combinations(range(n), 3))
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        
        # Precompute indices for k4s and k5s?
        # k4_indices[i] = (idx1, idx2, idx3, idx4)
        # k5_indices[i] = (idx1, ..., idx10)
        
        t_to_i = {t: i for i, t in enumerate(triples)}
        
        k4_idxs = []
        for quad in k4s:
            a,b,c,d = quad
            k4_idxs.append((
                t_to_i[(a,b,c)], t_to_i[(a,b,d)],
                t_to_i[(a,c,d)], t_to_i[(b,c,d)]
            ))
            
        k5_idxs = []
        for quint in k5s:
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(t_to_i[triple])
            k5_idxs.append(tuple(idxs))

        # Now verify is just checking arrays
        def super_fast_verify(colors):
            # Check Red K4
            for idxs in k4_idxs:
                if colors[idxs[0]] == 0 and colors[idxs[1]] == 0 and colors[idxs[2]] == 0 and colors[idxs[3]] == 0:
                    return False
            # Check Blue K5
            for idxs in k5_idxs:
                # Check if all 10 are 1
                # Unroll for speed?
                if (colors[idxs[0]] == 1 and colors[idxs[1]] == 1 and colors[idxs[2]] == 1 and
                    colors[idxs[3]] == 1 and colors[idxs[4]] == 1 and colors[idxs[5]] == 1 and
                    colors[idxs[6]] == 1 and colors[idxs[7]] == 1 and colors[idxs[8]] == 1 and
                    colors[idxs[9]] == 1):
                    return False
            return True

        # Search
        for M in [2, 3, 4, 5, 6, 7, 8]:
            if time.time() > time_limit:
                return None
            for L1 in range(M):
                for L2 in range(M):
                    for L3 in range(M):
                        # Skip trivial all same?
                        # Generate base residues
                        # residues[i] = (L1*a + L2*b + L3*c) % M for triple i
                        residues = np.array([(L1*a + L2*b + L3*c) % M for a,b,c in triples], dtype=np.int8)
                        
                        # Iterate over subsets of residues to be Red (0)
                        # Number of subsets is 2^M
                        # We can represent subset as a bitmask
                        for mask in range(1 << M):
                            # Check time
                            if time.time() > time_limit:
                                return None
                            
                            # Determine colors: 0 if residue is in mask, else 1
                            # mask bit r is 1 if residue r is Red (0)
                            # We want colors[i] = 0 if (mask >> residues[i]) & 1 else 1
                            
                            # Vectorized color generation
                            # Create a lookup table for color based on residue
                            color_lut = np.zeros(M, dtype=np.int8)
                            for r in range(M):
                                if (mask >> r) & 1:
                                    color_lut[r] = 0
                                else:
                                    color_lut[r] = 1
                                    
                            colors = color_lut[residues]
                            
                            if super_fast_verify(colors):
                                return colors.tolist()
        return None

    def local_search(n, time_limit, seed=0):
        """Fallback local search for smaller n or if algebraic fails."""
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        t_to_i = {t: i for i, t in enumerate(triples)}
        
        k4_idxs = []
        for quad in combinations(range(n), 4):
            a,b,c,d = quad
            k4_idxs.append((
                t_to_i[(a,b,c)], t_to_i[(a,b,d)],
                t_to_i[(a,c,d)], t_to_i[(b,c,d)]
            ))
        k5_idxs = []
        for quint in combinations(range(n), 5):
            idxs = [t_to_i[t] for t in combinations(quint, 3)]
            k5_idxs.append(tuple(idxs))
            
        np.random.seed(seed)
        colors = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
        
        def count_violations(cols):
            red = 0
            for idxs in k4_idxs:
                if cols[idxs[0]] == 0 and cols[idxs[1]] == 0 and cols[idxs[2]] == 0 and cols[idxs[3]] == 0:
                    red += 1
            blue = 0
            for idxs in k5_idxs:
                if (cols[idxs[0]] == 1 and cols[idxs[1]] == 1 and cols[idxs[2]] == 1 and
                    cols[idxs[3]] == 1 and cols[idxs[4]] == 1 and cols[idxs[5]] == 1 and
                    cols[idxs[6]] == 1 and cols[idxs[7]] == 1 and cols[idxs[8]] == 1 and
                    cols[idxs[9]] == 1):
                    blue += 1
            return red + blue

        current_cost = count_violations(colors)
        best_cost = current_cost
        best_colors = colors.copy()
        
        temp = 1.0
        cool_rate = 0.999
        min_temp = 0.01
        
        while time.time() < time_limit:
            t = np.random.randint(num_triples)
            old = colors[t]
            new = 1 - old
            colors[t] = new
            new_cost = count_violations(colors)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_colors = colors.copy()
                    if best_cost == 0:
                        return best_colors.tolist()
            else:
                # Revert
                colors[t] = old
                if temp > min_temp:
                    if np.random.rand() < np.exp(-(new_cost - current_cost) / temp):
                        colors[t] = new
                        current_cost = new_cost
                temp *= cool_rate
        
        return best_colors.tolist()

    final_n = 0
    final_colors = []

    # Try algebraic first for high n
    for n in [34, 33, 32, 31, 30, 29, 28]:
        if time.time() > deadline - 10:
            break
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # Give algebraic search a generous budget, it's fast per candidate
        # But if M is large, 2^M subsets can be many.
        # For n=34, verify is ~0.1s.
        # M=8: 8^3 * 2^8 = 512 * 256 = 131k candidates. 131k * 0.1s = 13100s. Too slow.
        # We need to prune or limit M.
        # Let's limit M to 6 for n=34? 6^3 * 2^6 = 216 * 64 = 13824. 13824 * 0.1s = 1382s. Still slow.
        # We need faster verify or fewer candidates.
        
        # Optimization: Only check a random subset of K4/K5s in super_fast_verify?
        # No, that's probabilistic.
        
        # Better: Check constraints incrementally during generation?
        # Or just reduce M to 5 for n=34?
        # M=5: 125 * 32 = 4000. 4000 * 0.1s = 400s. Still too slow for 115s budget.
        
        # We must make verify faster.
        # Using numpy for the check?
        # k4_idxs is a list of tuples.
        # We can convert k4_idxs to a numpy array of shape (num_k4, 4).
        # Then colors[k4_idxs] gives an array of shape (num_k4, 4).
        # Check if any row is all 0s: np.all(np.any(colors[k4_idxs] == 0, axis=1), axis=0)? No.
        # np.all(colors[k4_idxs] == 0, axis=1).any()
        
        # Let's refactor super_fast_verify to use numpy.
        
        def numpy_verify(colors_np):
            # colors_np is 1D array of int8
            # k4_idxs_np is 2D array of indices
            # k5_idxs_np is 2D array of indices
            
            # Red K4: all 4 indices have color 0
            red_violations = np.all(colors_np[k4_idxs_np] == 0, axis=1)
            if np.any(red_violations):
                return False
            
            # Blue K5: all 10 indices have color 1
            blue_violations = np.all(colors_np[k5_idxs_np] == 1, axis=1)
            if np.any(blue_violations):
                return False
            
            return True

        # Precompute numpy arrays for this n
        k4_idxs_np = np.array(k4_idxs, dtype=np.int32)
        k5_idxs_np = np.array(k5_idxs, dtype=np.int32)
        
        # Re-define super_fast_verify to use numpy
        # We need to pass k4_idxs_np and k5_idxs_np to it.
        # Since we are in a loop, we define it inside or capture it.
        
        def verify_np(colors_np):
            red_violations = np.all(colors_np[k4_idxs_np] == 0, axis=1)
            if np.any(red_violations):
                return False
            blue_violations = np.all(colors_np[k5_idxs_np] == 1, axis=1)
            if np.any(blue_violations):
                return False
            return True

        # Now verify is very fast. ~1ms?
        # 4000 candidates * 1ms = 4s. Feasible.
        # 13824 candidates * 1ms = 14s. Feasible.
        # 131k candidates * 1ms = 131s. Borderline.
        
        # So we can try M up to 8 if time allows.
        
        found = try_algebraic(n, time.time() + min(20.0, remaining * 0.5))
        if found:
            # Final verification to be safe
            if verify(n, found):
                final_n = n
                final_colors = found
                break

    # If algebraic failed, try local search for smaller n
    if final_n == 0:
        for n in [28, 27, 26, 25, 24, 23, 22, 21, 20]:
            if time.time() > deadline - 5:
                break
            remaining = deadline - time.time()
            if remaining < 2:
                break
            
            # Local search is slow, give it limited time
            t_limit = time.time() + min(10.0, remaining * 0.8)
            colors = local_search(n, t_limit, seed=42)
            
            if verify(n, colors):
                final_n = n
                final_colors = colors
                break

    # Fallback to n=9
    if final_n == 0:
        n = 9
        triples = list(combinations(range(n), 3))
        colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
        if verify(n, colors):
            final_n = n
            final_colors = colors
        else:
            final_n = 1
            final_colors = [0]

    return {'n': final_n, 'colors': final_colors}
