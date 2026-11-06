"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    max_target = 128
    
    # Generate all 2,3,5-smooth numbers up to max_target
    smooth = []
    for n in range(1, max_target + 1):
        x = n
        while x % 2 == 0:
            x //= 2
        while x % 3 == 0:
            x //= 3
        while x % 5 == 0:
            x //= 5
        if x == 1:
            smooth.append(n)
    
    # For each target, find the minimum number of terms in a valid antichain representation
    answers = [None] * max_target
    
    # Precompute: for each smooth number, which other smooth numbers does it divide?
    # smooth_divides[i] = set of indices j such that smooth[i] divides smooth[j] and i != j
    n_smooth = len(smooth)
    smooth_divides = [set() for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j and smooth[j] % smooth[i] == 0:
                smooth_divides[i].add(j)
    
    # For each target, try increasing term counts
    for n in range(1, max_target + 1):
        if time.time() > deadline:
            # Return best effort: use seed-like fallback
            break
        
        candidates = [i for i in range(n_smooth) if smooth[i] <= n]
        
        found = False
        for count in range(1, 31):
            if time.time() > deadline:
                break
            
            # Try to find an antichain of size 'count' that sums to n
            # Use combinations with pruning
            best_combo = None
            
            # For small counts, use combinations directly
            if count <= 4:
                for combo in combinations(candidates, count):
                    # Check sum
                    if sum(smooth[i] for i in combo) != n:
                        continue
                    # Check antichain: no term divides another
                    valid = True
                    for a_idx in range(len(combo)):
                        for b_idx in range(a_idx + 1, len(combo)):
                            a = combo[a_idx]
                            b = combo[b_idx]
                            if smooth[b] % smooth[a] == 0 or smooth[a] % smooth[b] == 0:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        best_combo = combo
                        break
            else:
                # For larger counts, use a more efficient search
                # This is harder; try a greedy/backtracking approach
                # For now, skip large counts if we've already found a solution
                # or use a heuristic
                pass
            
            if best_combo is not None:
                answers[n - 1] = [smooth[i] for i in best_combo]
                found = True
                break
        
        if not found:
            # Fallback: use a simple valid representation if we didn't find one
            # This shouldn't happen for small targets, but just in case
            # Try to find any valid antichain representation
            for count in range(1, 31):
                for combo in combinations(candidates, count):
                    if sum(smooth[i] for i in combo) != n:
                        continue
                    valid = True
                    for a_idx in range(len(combo)):
                        for b_idx in range(a_idx + 1, len(combo)):
                            a = combo[a_idx]
                            b = combo[b_idx]
                            if smooth[b] % smooth[a] == 0 or smooth[a] % smooth[b] == 0:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        answers[n - 1] = [smooth[i] for i in combo]
                        break
                if answers[n - 1] is not None:
                    break
    
    # If we ran out of time, fill in remaining with a simple heuristic
    for n in range(1, max_target + 1):
        if answers[n - 1] is None:
            # Simple fallback: try to build a valid representation
            # For now, just use the number itself if it's smooth
            if n in smooth:
                answers[n - 1] = [n]
            else:
                # Find any valid antichain
                candidates = [i for i in range(n_smooth) if smooth[i] <= n]
                for count in range(1, 31):
                    for combo in combinations(candidates, count):
                        if sum(smooth[i] for i in combo) != n:
                            continue
                        valid = True
                        for a_idx in range(len(combo)):
                            for b_idx in range(a_idx + 1, len(combo)):
                                a = combo[a_idx]
                                b = combo[b_idx]
                                if smooth[b] % smooth[a] == 0 or smooth[a] % smooth[b] == 0:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            answers[n - 1] = [smooth[i] for i in combo]
                            break
                    if answers[n - 1] is not None:
                        break
                if answers[n - 1] is None:
                    # Last resort: should not happen
                    answers[n - 1] = [n]
    
    return answers
# EVOLVE-BLOCK-END
