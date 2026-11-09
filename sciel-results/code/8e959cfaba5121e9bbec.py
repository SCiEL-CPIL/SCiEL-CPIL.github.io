"""Rewritten program: pack distinct multiples of a fixed integer set using vectorized feasibility check."""
# EVOLVE-BLOCK-START
import numpy as np

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
LCM = 30030

def _find_optimal_window():
    """
    Find the minimum window length L and a valid start S.
    Uses vectorized checks over S in one LCM period.
    """
    # Precompute residues for each divisor for all S in 0..LCM-1
    # S is the start of the window [S, S+L]
    # For a given S and divisor d, a multiple exists in [S, S+L] iff:
    # Let r = S % d. The next multiple is at S + (d - r) % d.
    # This offset is <= L.
    
    for L in range(9, 200): # Search up to L=199
        # Check all S in 0..LCM-1
        # We can vectorize this over S
        S_arr = np.arange(LCM)
        
        valid_S = np.ones(LCM, dtype=bool)
        candidates_list = []
        
        for d in DIVISORS:
            # Compute offset to next multiple for each S
            r = S_arr % d
            offset = (d - r) % d
            # Check if offset <= L
            has_multiple = offset <= L
            valid_S &= has_multiple
            
            # If valid, we need to store the candidate multiple(s)
            # For distinctness check, we need the actual values.
            # If L is small, usually only 1 candidate.
            # If L >= d, there might be 2.
            
            # For now, just mark validity.
            # We will refine distinctness check after finding candidate S.
            
        # Find S where valid_S is True
        candidate_S_indices = np.where(valid_S)[0]
        
        if len(candidate_S_indices) == 0:
            continue
            
        # For each candidate S, check distinctness
        # To speed up, we can check all at once?
        # Distinctness is hard to vectorize directly because of the "choice" aspect if multiple candidates exist.
        # However, for small L, most divisors have only 1 candidate.
        
        for S in candidate_S_indices:
            # Reconstruct candidates for this S
            mults = []
            distinct = True
            seen = set()
            for d in DIVISORS:
                r = S % d
                offset = (d - r) % d
                m = S + offset
                # Check if m is in [S, S+L]
                if m <= S + L:
                    if m in seen:
                        distinct = False
                        break
                    seen.add(m)
                    mults.append(m)
                else:
                    # This shouldn't happen if valid_S was True, 
                    # unless there are multiple candidates and the first one is invalid?
                    # No, offset is the *next* multiple. If offset > L, no multiple.
                    # But valid_S ensured offset <= L.
                    # Wait, if L >= d, there is a second multiple at m+d.
                    # The logic above only checks the *first* multiple.
                    # If the first multiple is taken by another divisor, we might need the second.
                    # But for small L (e.g., < 10), L < d for all d except d=6.
                    # For d=6, L=9, there are 2 multiples.
                    
                    # So we need to handle multiple candidates.
                    # Let's collect all candidates in [S, S+L].
                    pass
            
            # Let's redo the candidate collection properly for this S
            mults = []
            seen = set()
            possible = True
            for d in DIVISORS:
                start_k = (S + d - 1) // d
                m = start_k * d
                cands = []
                while m <= S + L:
                    cands.append(m)
                    m += d
                if not cands:
                    possible = False
                    break
                mults.append(cands)
            
            if not possible:
                continue
                
            # Backtrack to find distinct assignment
            # Since L is small, cands are small lists
            assignment = [None] * 10
            used = set()
            
            def backtrack(idx):
                if idx == 10:
                    return True
                for m in mults[idx]:
                    if m not in used:
                        used.add(m)
                        assignment[idx] = m
                        if backtrack(idx + 1):
                            return True
                        used.remove(m)
                        assignment[idx] = None
                return False

            if backtrack(0):
                return assignment, L

    # Fallback
    return DIVISORS[:], max(DIVISORS) - min(DIVISORS)

_optimal_solution = None

def _get_solution():
    global _optimal_solution
    if _optimal_solution is None:
        _optimal_solution = _find_optimal_window()
    return _optimal_solution

def construct():
    sol, _ = _get_solution()
    return [int(x) for x in sol]

# EVOLVE-BLOCK-END
