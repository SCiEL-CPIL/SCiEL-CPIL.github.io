import time
import numpy as np
from math import lcm, gcd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import sparse
from fractions import Fraction

def _omega(n):
    """Total number of prime factors (with multiplicity)."""
    count = 0
    d = 2
    while d * d <= n:
        while n % d == 0:
            count += 1
            n //= d
        d += 1
    if n > 1:
        count += 1
    return count

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 90s kill

    # 1. Define the set of candidates
    candidates = list(range(2, 257))
    n = len(candidates)
    
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # 2. Precompute Omega parity for partitioning
    omega_parity = [_omega(v) % 2 for v in candidates]
    
    # Partition into even-Omega and odd-Omega
    even_idx = [i for i in range(n) if omega_parity[i] == 0]
    odd_idx = [i for i in range(n) if omega_parity[i] == 1]
    
    # 3. Construct the full conflict graph
    rows = []
    cols = []
    data = []
    
    conflict_count = 0
    for i in range(n):
        vi = candidates[i]
        for j in range(i + 1, n):
            vj = candidates[j]
            g = gcd(vi, vj)
            l = vi * vj // g
            if l <= 256:
                rows.append(conflict_count)
                cols.append(i)
                data.append(1)
                
                rows.append(conflict_count)
                cols.append(j)
                data.append(1)
                
                conflict_count += 1
                
    m = conflict_count
    if m > 0:
        A_full = sparse.csr_matrix((data, (rows, cols)), shape=(m, n))
    else:
        A_full = sparse.csr_matrix((0, n))
        
    # 4. Objective function
    c = np.array([-1.0 / v for v in candidates])
    
    # 5. First, try solving the full MIP with a tight time limit
    best_selected = None
    best_score = 0.0
    
    def score_of(selected):
        """Compute exact reciprocal sum using Fraction."""
        s = Fraction(0, 1)
        for v in selected:
            s += Fraction(1, v)
        return s
    
    def is_valid(selected):
        """Check if selected is a valid independent set."""
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                if lcm(selected[i], selected[j]) <= 256:
                    return False
        return True
    
    # Try full MIP
    if time.time() - start_time < deadline - 10:
        try:
            constraints = LinearConstraint(A_full, ub=np.ones(m))
            bounds = Bounds(lb=0, ub=1)
            integrality = np.ones(n)
            
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, options={"time_limit": 20})
            
            if res.success:
                x_sol = res.x
                selected = [candidates[i] for i in range(n) if x_sol[i] > 0.5]
                if is_valid(selected) and len(selected) > 0:
                    best_selected = selected
                    best_score = score_of(selected)
        except Exception:
            pass
    
    # 6. If full MIP didn't give good results, try partition-based approach
    if time.time() - start_time < deadline - 5:
        for part_name, part_indices in [("even", even_idx), ("odd", odd_idx)]:
            if time.time() - start_time >= deadline - 3:
                break
            
            k = len(part_indices)
            if k == 0:
                continue
            
            # Build sub-MIP for this partition
            sub_c = c[part_indices]
            
            # Extract sub-conflict matrix
            if m > 0:
                A_sub = A_full[:, part_indices]
                A_sub_csr = A_sub.tocsr()
                # Remove empty rows
                A_sub_csr.sum_duplicates()
                # Get non-empty rows
                row_sums = np.asarray(A_sub_csr.sum(axis=1)).flatten()
                non_empty = row_sums > 0
                if non_empty.any():
                    A_sub_final = A_sub_csr[non_empty, :]
                else:
                    A_sub_final = sparse.csr_matrix((0, k))
            else:
                A_sub_final = sparse.csr_matrix((0, k))
            
            m_sub = A_sub_final.shape[0]
            
            if m_sub > 0:
                try:
                    sub_constraints = LinearConstraint(A_sub_final, ub=np.ones(m_sub))
                    sub_bounds = Bounds(lb=0, ub=1)
                    sub_integrality = np.ones(k)
                    
                    res_sub = milp(c=sub_c, constraints=sub_constraints, integrality=sub_integrality, bounds=sub_bounds, options={"time_limit": 8})
                    
                    if res_sub.success:
                        x_sub = res_sub.x
                        sub_selected = [part_indices[i] for i in range(k) if x_sub[i] > 0.5]
                        sub_vals = [candidates[i] for i in sub_selected]
                        
                        if is_valid(sub_vals) and len(sub_vals) > 0:
                            sub_score = score_of(sub_vals)
                            if sub_score > best_score:
                                best_score = sub_score
                                best_selected = sub_vals
                except Exception:
                    pass
            else:
                # No conflicts in this partition, take all
                part_vals = [candidates[i] for i in part_indices]
                if is_valid(part_vals):
                    part_score = score_of(part_vals)
                    if part_score > best_score:
                        best_score = part_score
                        best_selected = part_vals
    
    # 7. Local search polish: try to improve by swapping
    if best_selected is not None and time.time() - start_time < deadline - 2:
        selected = list(best_selected)
        selected_set = set(selected)
        current_score = best_score
        
        improved = True
        while improved and time.time() - start_time < deadline - 1:
            improved = False
            
            # Try adding each unselected candidate
            for v in candidates:
                if v in selected_set:
                    continue
                # Check if v can be added
                if all(lcm(v, y) > 256 for y in selected):
                    new_selected = selected + [v]
                    new_score = current_score + Fraction(1, v)
                    if new_score > current_score:
                        selected = new_selected
                        selected_set.add(v)
                        current_score = new_score
                        improved = True
            
            # Try removing low-value candidates and adding multiple high-value ones
            # Sort by value ascending (low reciprocal weight first)
            for v in sorted(selected, key=lambda x: x):  # ascending value = low weight
                if len(selected) <= 1:
                    break
                # Try removing v
                remaining = [x for x in selected if x != v]
                remaining_set = set(remaining)
                removed_score = current_score - Fraction(1, v)
                
                # Try adding candidates not in remaining that don't conflict
                added = 0
                new_selected = list(remaining)
                new_score = removed_score
                
                for u in candidates:
                    if u in remaining_set or u == v:
                        continue
                    if all(lcm(u, y) > 256 for y in new_selected):
                        new_selected.append(u)
                        remaining_set.add(u)
                        new_score += Fraction(1, u)
                        added += 1
                        if added >= 2:  # Only worth it if we add at least 2
                            break
                
                if added >= 2 and new_score > current_score:
                    selected = new_selected
                    selected_set = set(selected)
                    current_score = new_score
                    improved = True
    
    # 8. Final fallback if nothing worked
    if best_selected is None or len(best_selected) == 0:
        # Greedy fallback
        selected = []
        for v in sorted(candidates):
            if all(lcm(v, y) > 256 for y in selected):
                selected.append(v)
        best_selected = selected
    
    # Final verification
    if not is_valid(best_selected):
        # Emergency fallback
        selected = []
        for v in sorted(candidates):
            if all(lcm(v, y) > 256 for y in selected):
                selected.append(v)
        best_selected = selected
    
    return best_selected
