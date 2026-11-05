import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safe margin before 150s kill

    def get_triple_indices(n):
        """Return list of all triples (a,b,c) with a<b<c."""
        return list(combinations(range(n), 3))

    def verify(n, colors):
        """Verify coloring has no red K4 and no blue K5."""
        triples = get_triple_indices(n)
        triple_to_idx = {t: i for i, t in enumerate(triples)}

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

    def count_violations(n, colors):
        """Count red K4s and blue K5s."""
        triples = get_triple_indices(n)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        red_count = 0
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if (colors[triple_to_idx[(a, b, c)]] == 0 and
                colors[triple_to_idx[(a, b, d)]] == 0 and
                colors[triple_to_idx[(a, c, d)]] == 0 and
                colors[triple_to_idx[(b, c, d)]] == 0):
                red_count += 1
                
        blue_count = 0
        for quint in combinations(range(n), 5):
            all_blue = True
            for triple in combinations(quint, 3):
                if colors[triple_to_idx[triple]] != 1:
                    all_blue = False
                    break
            if all_blue:
                blue_count += 1
                
        return red_count + blue_count

    def generate_field_coloring(n, p, coeffs, subset_type, subset_param):
        """
        Generate coloring based on linear form over GF(p) or Z/pZ.
        coeffs: tuple (c1, c2, c3) for c1*a + c2*b + c3*c
        subset_type: 'residue' (QR), 'interval' (0..k), 'parity'
        """
        triples = get_triple_indices(n)
        colors = []
        
        # Precompute QR set if needed
        qr_set = None
        if subset_type == 'residue':
            qr_set = set((x*x) % p for x in range(p))
            if 0 in qr_set:
                qr_set.add(0) # Ensure 0 is in red if we treat QR as 1, but usually 0 is separate. 
                # Let's define Red if val in QR_set.
        
        for a, b, c in triples:
            val = (coeffs[0]*a + coeffs[1]*b + coeffs[2]*c) % p
            
            if subset_type == 'residue':
                is_red = 1 if val in qr_set else 0
            elif subset_type == 'interval':
                # Red if val <= subset_param
                is_red = 1 if val <= subset_param else 0
            elif subset_type == 'parity':
                is_red = val % 2
            else:
                is_red = 0
                
            # We want to return 0 for Red, 1 for Blue
            # So if is_red is 1, color is 0. If is_red is 0, color is 1.
            colors.append(1 - is_red)
            
        return colors

    def try_construction(n):
        """Try algebraic constructions for given n."""
        best_colors = None
        best_violations = float('inf')
        
        # Primes to try
        primes = [31, 37, 41, 43]
        
        # Coefficients to try (non-zero)
        coeffs_list = [
            (1, 1, 1),
            (1, 1, -1),
            (1, -1, 1),
            (-1, 1, 1),
            (1, 2, 1),
            (1, 1, 2),
            (2, 1, 1),
            (1, 2, 3),
            (1, 3, 2),
            (2, 3, 1)
        ]
        
        for p in primes:
            if p < n:
                continue
                
            for coeffs in coeffs_list:
                # Normalize coeffs to avoid duplicates (e.g., multiply by inverse of first non-zero)
                # For simplicity, we just try the list.
                
                for subset_type in ['residue', 'interval']:
                    if subset_type == 'residue':
                        # Try with and without 0 in QR set? 
                        # Standard QR includes 0? No, 0 is not a QR usually. 
                        # Let's try two variants: QR set and QR set + {0}
                        pass 
                        
                        # Variant 1: Red if val is QR (non-zero)
                        # Variant 2: Red if val is QR or 0
                        
                        # We can just iterate subset_param for interval, and fixed for residue
                        # Let's add a 'residue0' type
                        for sub_variant in [0, 1]:
                            if time.time() > deadline - 2:
                                return None, float('inf')
                                
                            # Adjust subset_type logic inside generate? 
                            # Easier to just call a specific generator logic here
                            
                            triples = get_triple_indices(n)
                            colors = []
                            
                            qr_set = set((x*x) % p for x in range(1, p))
                            if sub_variant == 1:
                                qr_set.add(0)
                                
                            for a, b, c in triples:
                                val = (coeffs[0]*a + coeffs[1]*b + coeffs[2]*c) % p
                                is_red = 1 if val in qr_set else 0
                                colors.append(1 - is_red)
                                
                            v = count_violations(n, colors)
                            if v < best_violations:
                                best_violations = v
                                best_colors = colors
                                if v == 0:
                                    return colors, 0

                    elif subset_type == 'interval':
                        # Try different interval sizes
                        # Heuristic: balanced colors are best
                        # Number of triples is C(n,3).
                        # If p is large, val is uniform.
                        # Try k such that (k+1)/p * TotalTriples is approx half
                        total_triples = len(get_triple_indices(n))
                        half = total_triples / 2
                        # Expected red count = (k+1)/p * total_triples
                        # k+1 = half * p / total_triples
                        k_est = int((half * p) / total_triples) - 1
                        k_est = max(0, min(p-1, k_est))
                        
                        for k in range(max(0, k_est-5), min(p, k_est+6)):
                            if time.time() > deadline - 2:
                                return None, float('inf')
                                
                            triples = get_triple_indices(n)
                            colors = []
                            for a, b, c in triples:
                                val = (coeffs[0]*a + coeffs[1]*b + coeffs[2]*c) % p
                                is_red = 1 if val <= k else 0
                                colors.append(1 - is_red)
                                
                            v = count_violations(n, colors)
                            if v < best_violations:
                                best_violations = v
                                best_colors = colors
                                if v == 0:
                                    return colors, 0

        return best_colors, best_violations

    final_n = 0
    final_colors = []

    # Try to find a valid coloring for n from 34 down to 1
    # We prioritize higher n
    for n in range(34, 0, -1):
        if time.time() > deadline - 5:
            break

        if n <= final_n:
            continue

        # Try algebraic construction
        colors, violations = try_construction(n)
        
        if colors is not None and violations == 0:
            # Verify to be safe
            if verify(n, colors):
                final_n = n
                final_colors = colors
                break
        elif colors is not None and violations < 50:
            # If close, we might try a quick local repair on this specific high-quality start
            # But for this structural change, we stick to the algebraic result if valid.
            # If invalid, we continue to next n or try to repair.
            # Let's add a very short local repair for high-quality algebraic starts
            pass

    # Fallback to seed behavior if algebraic fails or finds nothing
    if final_n == 0:
        # Re-use the seed's local search as a fallback for lower n or if algebraic fails
        # But we must keep the function signature.
        # Let's just run the original repair_search logic for n=34..10 as a fallback
        # To save space, I will implement a simplified version of the repair search here
        
        def simple_repair(n, time_limit, seed=0):
            triples = get_triple_indices(n)
            num_triples = len(triples)
            np.random.seed(seed)
            
            # Simple random init
            colors = np.random.randint(0, 2, num_triples).astype(np.int8)
            
            # Precompute K4s and K5s
            triple_to_idx = {t: i for i, t in enumerate(triples)}
            k4s = []
            for quad in combinations(range(n), 4):
                a,b,c,d = quad
                k4s.append([triple_to_idx[(a,b,c)], triple_to_idx[(a,b,d)], triple_to_idx[(a,c,d)], triple_to_idx[(b,c,d)]])
            
            k5s = []
            for quint in combinations(range(n), 5):
                idxs = [triple_to_idx[t] for t in combinations(quint, 3)]
                k5s.append(idxs)
                
            triple_to_k4s = [[] for _ in range(num_triples)]
            triple_to_k5s = [[] for _ in range(num_triples)]
            for i, idxs in enumerate(k4s):
                for t in idxs:
                    triple_to_k4s[t].append(i)
            for i, idxs in enumerate(k5s):
                for t in idxs:
                    triple_to_k5s[t].append(i)
                    
            def count_v():
                rc = sum(1 for idxs in k4s if all(colors[i]==0 for i in idxs))
                bc = sum(1 for idxs in k5s if all(colors[i]==1 for i in idxs))
                return rc+bc
                
            current_cost = count_v()
            best_cost = current_cost
            best_colors = colors.copy()
            
            if current_cost == 0:
                return colors.tolist(), True
                
            temp = 1.0
            while time.time() < time_limit:
                t = np.random.randint(num_triples)
                old_c = colors[t]
                new_c = 1 - old_c
                
                # Calculate delta
                delta = 0
                for k4_i in triple_to_k4s[t]:
                    idxs = k4s[k4_i]
                    rc = sum(1 for i in idxs if colors[i]==0)
                    if old_c == 0:
                        if rc == 4: delta -= 1
                    else:
                        if rc == 3: delta += 1
                        
                for k5_i in triple_to_k5s[t]:
                    idxs = k5s[k5_i]
                    bc = sum(1 for i in idxs if colors[i]==1)
                    if old_c == 1:
                        if bc == 10: delta -= 1
                    else:
                        if bc == 9: delta += 1
                        
                new_cost = current_cost + delta
                accept = delta <= 0 or (temp > 0.01 and np.random.rand() < np.exp(-delta/temp))
                
                if accept:
                    colors[t] = new_c
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors.copy()
                        if best_cost == 0:
                            return best_colors.tolist(), True
                temp *= 0.9999
                
            return best_colors.tolist(), (best_cost == 0)

        for n in range(34, 0, -1):
            if time.time() > deadline - 2:
                break
            if n <= final_n:
                continue
            remaining = deadline - time.time()
            if remaining < 2:
                break
            time_budget = min(10.0, remaining * 0.5)
            if time_budget < 1:
                break
            
            for seed in range(5):
                if time.time() > deadline - 1:
                    break
                try:
                    cols, is_valid = simple_repair(n, time.time() + time_budget/5, seed=seed)
                    if is_valid:
                        if verify(n, cols):
                            final_n = n
                            final_colors = cols
                            break
                except:
                    continue
            if final_n == n:
                break

    # Absolute Fallback
    if final_n == 0:
        n = 9
        triples = get_triple_indices(n)
        colors = [0 if (a+b+c)%3==0 else 1 for a,b,c in triples]
        if verify(n, colors):
            final_n = n
            final_colors = colors
        else:
            final_n = 1
            final_colors = [0]

    return {'n': final_n, 'colors': final_colors}
