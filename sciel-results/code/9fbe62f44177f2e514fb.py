"""Rewritten program for divisibility-antichain representations by smooth numbers."""
# EVOLVE-BLOCK-START
import sys
from itertools import combinations

def _generate_smooth(max_val):
    """Generate all 2^a * 3^b * 5^c numbers up to max_val."""
    smooth = set()
    a = 1
    while a <= max_val:
        b = a
        while b <= max_val:
            c = b
            while c <= max_val:
                smooth.add(c)
                c *= 5
            b *= 3
        a *= 2
    return sorted(smooth)

def construct():
    """
    Construct 128 rows for targets 1..128.
    Each row contains 1..30 distinct positive 2^a*3^b*5^c terms summing to target,
    forming a divisibility antichain.
    Minimize total number of summands over all targets.
    """
    max_target = 128
    smooth = _generate_smooth(max_target)
    smooth_set = set(smooth)
    n_smooth = len(smooth)
    
    # Precompute divisibility matrix
    # divs[i][j] = 1 if smooth[i] divides smooth[j]
    divs = [[0] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j and smooth[j] % smooth[i] == 0:
                divs[i][j] = 1
    
    # Map value to index for quick lookup? Not needed if we use indices in combinations.
    
    answers = []
    
    for n in range(1, max_target + 1):
        # Filter smooth numbers <= n
        valid_indices = [i for i, v in enumerate(smooth) if v <= n]
        
        found = None
        # Try k from 1 to 10 (should be enough for 128)
        for k in range(1, 11):
            if k > len(valid_indices):
                break
            
            # Generate combinations of indices of size k
            # To optimize, we can iterate, but itertools is fine for small k
            for combo in combinations(valid_indices, k):
                # Check sum
                s = sum(smooth[i] for i in combo)
                if s != n:
                    continue
                
                # Check antichain
                is_anti = True
                for a_idx in range(len(combo)):
                    i = combo[a_idx]
                    for b_idx in range(a_idx + 1, len(combo)):
                        j = combo[b_idx]
                        if divs[i][j] or divs[j][i]:
                            is_anti = False
                            break
                    if not is_anti:
                        break
                
                if is_anti:
                    found = [smooth[i] for i in combo]
                    break
            
            if found is not None:
                break
        
        if found is None:
            # Fallback, though unlikely for 1..128
            # Try larger k or just return [n] if smooth?
            # The problem guarantees a solution exists with <= 30 terms.
            # If k > 10, we might need to search further, but let's assume k<=10 is enough.
            # For safety, if not found, we can do a deeper search or just return a valid one if we find one later.
            # Let's extend search to 20 just in case.
            for k in range(11, 21):
                if k > len(valid_indices):
                    break
                for combo in combinations(valid_indices, k):
                    s = sum(smooth[i] for i in combo)
                    if s != n:
                        continue
                    is_anti = True
                    for a_idx in range(len(combo)):
                        i = combo[a_idx]
                        for b_idx in range(a_idx + 1, len(combo)):
                            j = combo[b_idx]
                            if divs[i][j] or divs[j][i]:
                                is_anti = False
                                break
                        if not is_anti:
                            break
                    if is_anti:
                        found = [smooth[i] for i in combo]
                        break
                if found is not None:
                    break
        
        if found is None:
            # Extremely unlikely fallback
            found = [n] if n in smooth_set else [1]
            
        answers.append(found)
    
    return answers

# EVOLVE-BLOCK-END
