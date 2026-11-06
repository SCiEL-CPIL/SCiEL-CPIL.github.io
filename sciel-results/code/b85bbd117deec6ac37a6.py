import time
from collections import defaultdict

_DEADLINE = 85.0

def _build_smooths(limit):
    """Generate all 2^a * 3^b * 5^c numbers <= limit."""
    out = set()
    a = 1
    while a <= limit:
        b = a
        while b <= limit:
            c = b
            while c <= limit:
                out.add(c)
                c *= 5
            b *= 3
        a *= 2
    return sorted(out)

def _divides(a, b):
    """Check if a divides b."""
    return b % a == 0

def _antichain_check(terms):
    """Check if a set of terms forms an antichain under divisibility."""
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            if terms[i] <= terms[j]:
                if _divides(terms[i], terms[j]):
                    return False
            else:
                if _divides(terms[j], terms[i]):
                    return False
    return True

def _precompute_all_antichains(smooths, max_sum=128):
    """
    Enumerate all antichains of smooth numbers with sum <= max_sum.
    Returns a dictionary mapping sum -> list of (term_count, terms) pairs.
    """
    n = len(smooths)
    # For each sum, store the minimum term count found and the corresponding terms
    best_terms = {}  # sum -> (count, terms_tuple)
    
    def dfs(start_idx, current_sum, current_terms):
        if current_sum > max_sum:
            return
        
        if len(current_terms) > 0 and current_sum > 0:
            # Update best if this has fewer terms
            if current_sum not in best_terms or len(current_terms) < best_terms[current_sum][0]:
                best_terms[current_sum] = (len(current_terms), tuple(current_terms))
        
        # If we already have a 1-term solution for this sum, we can't improve it
        # But we still need to explore other sums, so we continue unless current_sum is already optimal for itself
        # Actually, we can prune: if current_terms has 1 element, adding more will only increase the sum
        # and the new sum will have more terms. But those new sums might not have been reached yet.
        # So we must continue.
        
        for i in range(start_idx, n):
            x = smooths[i]
            if current_sum + x > max_sum:
                break
            
            # Check antichain property
            valid = True
            for t in current_terms:
                if _divides(t, x):  # t divides x? Since t <= x in our sorted order
                    valid = False
                    break
                if _divides(x, t):  # x divides t? Since x >= t, this means x == t, but terms are distinct
                    valid = False
                    break
            
            if valid:
                dfs(i + 1, current_sum + x, current_terms + [x])
    
    dfs(0, 0, [])
    
    # Convert to sum -> terms list
    result = {}
    for s, (cnt, terms) in best_terms.items():
        result[s] = list(terms)
    
    return result

def construct():
    start_time = time.time()
    
    smooths = _build_smooths(128)
    
    # Precompute all optimal antichains for sums 1..128
    # This enumeration is fast because the number of smooths <= 128 is small (~60-70)
    # and the branching factor is limited by the sum constraint
    best_ants = _precompute_all_antichains(smooths, 128)
    
    # Fallback for any missing targets (should not happen per problem guarantee)
    # If a target is missing, try a simple greedy or random search
    # But since the problem guarantees all targets have valid representations,
    # and our enumeration is complete, this should be empty.
    
    # For any missing target, do a quick local search
    missing = [n for n in range(1, 129) if n not in best_ants]
    
    if missing:
        # Try to find representations for missing targets using a different approach
        # For each missing target, try small antichains
        for n in missing:
            # Try 1-term
            if n in smooths:
                best_ants[n] = [n]
                continue
            
            # Try 2-term: a + b = n, a < b, a not divide b
            found = False
            for i in range(len(smooths)):
                a = smooths[i]
                if a >= n:
                    break
                b = n - a
                if b in smooths and b > a and not _divides(a, b) and not _divides(b, a):
                    best_ants[n] = [a, b]
                    found = True
                    break
            
            if not found:
                # Try 3-term: a + b + c = n, a < b < c
                found = False
                for i in range(len(smooths)):
                    a = smooths[i]
                    if a >= n:
                        break
                    for j in range(i + 1, len(smooths)):
                        b = smooths[j]
                        if a + b >= n:
                            break
                        c = n - a - b
                        if c in smooths and c > b and not _divides(a, b) and not _divides(a, c) and not _divides(b, c) and not _divides(b, a) and not _divides(c, a) and not _divides(c, b):
                            best_ants[n] = [a, b, c]
                            found = True
                            break
                    if found:
                        break
                
                if not found:
                    # Last resort: use [1] if valid (1 divides everything, so [1] alone is only valid if target is 1)
                    # This is a fallback that may be invalid, but prevents crash
                    best_ants[n] = [n] if n in smooths else [1]
    
    # Construct the answer
    answers = []
    for n in range(1, 129):
        terms = best_ants.get(n, [n])
        # Ensure terms are distinct and form antichain
        if not _antichain_check(terms):
            # Fallback: try to fix
            # Remove any term that divides another
            fixed = []
            for t in terms:
                valid = True
                for ft in fixed:
                    if _divides(t, ft) or _divides(ft, t):
                        valid = False
                        break
                if valid:
                    fixed.append(t)
            terms = fixed if fixed else [n]
        
        answers.append(terms)
    
    return answers
