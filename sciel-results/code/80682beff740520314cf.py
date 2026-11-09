import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    period = 30030

    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s kill

    # Search for W from 1 upwards
    for W in range(1, 5000):
        if time.time() > deadline:
            break
        
        # Iterate L from 1 to period
        for L in range(1, period + 1):
            if time.time() > deadline:
                break
            
            # Generate candidates for each divisor
            candidates = []
            valid = True
            for d in divisors:
                k_start = (L + d - 1) // d
                m_start = k_start * d
                m_end = ((L + W) // d) * d
                
                if m_start > m_end:
                    valid = False
                    break
                
                num_cands = (m_end - m_start) // d + 1
                candidates.append((m_start, d, num_cands))
            
            if not valid:
                continue
            
            # Quick check: total candidates >= n
            total_cands = sum(c[2] for c in candidates)
            if total_cands < n:
                continue
            
            # Sort by number of candidates (ascending) - MRV heuristic
            order = sorted(range(n), key=lambda i: candidates[i][2])
            sorted_cands = [(i, candidates[i]) for i in order]
            
            # Backtracking with used set
            used = set()
            assignment = [None] * n
            
            def backtrack(idx):
                if time.time() > deadline:
                    return False
                if idx == n:
                    return True
                orig_idx, (m_start, d, num_cands) = sorted_cands[idx]
                for j in range(num_cands):
                    val = m_start + j * d
                    if val not in used:
                        used.add(val)
                        assignment[orig_idx] = val
                        if backtrack(idx + 1):
                            return True
                        used.remove(val)
                        assignment[orig_idx] = None
                return False
            
            if backtrack(0):
                result = [int(assignment[i]) for i in range(n)]
                return result
        
        if time.time() > deadline:
            break
    
    # Fallback: return the divisors themselves (valid but not optimal)
    return [int(d) for d in divisors]
