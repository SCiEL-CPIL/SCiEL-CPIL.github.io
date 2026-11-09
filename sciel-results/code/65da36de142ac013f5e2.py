"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    # 1. Generate smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    
    # 2. Precompute divisibility relations
    # divs[i][j] = True if smooth[i] divides smooth[j]
    n_smooth = len(smooth)
    divs = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i == j:
                divs[i][j] = True
            elif smooth[j] % smooth[i] == 0:
                divs[i][j] = True
    
    # 3. For each target, find minimal antichain representation
    answers = []
    for n in range(1, 129):
        if time.time() > deadline:
            # Return what we have so far, pad with None if needed
            while len(answers) < 128:
                answers.append(None)
            return answers
        
        # Candidate smooth numbers <= n
        candidates = [(idx, val) for idx, val in enumerate(smooth) if val <= n]
        cand_indices = [c[0] for c in candidates]
        cand_values = [c[1] for c in candidates]
        
        best_answer = None
        
        # Try 1 term first
        for idx, val in candidates:
            if val == n:
                best_answer = [val]
                break
        
        if best_answer is None and time.time() < deadline:
            # Try 2 terms
            for i in range(len(candidates)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(candidates)):
                    if cand_values[i] + cand_values[j] == n:
                        # Check antichain: neither divides the other
                        if not divs[cand_indices[i]][cand_indices[j]] and not divs[cand_indices[j]][cand_indices[i]]:
                            best_answer = [cand_values[i], cand_values[j]]
                            break
                if best_answer is not None:
                    break
        
        if best_answer is None and time.time() < deadline:
            # Try 3 terms
            for i in range(len(candidates)):
                if time.time() > deadline:
                    break
                vi = cand_values[i]
                ii = cand_indices[i]
                for j in range(i + 1, len(candidates)):
                    vj = cand_values[j]
                    if vi + vj >= n:
                        break
                    jj = cand_indices[j]
                    if divs[ii][jj] or divs[jj][ii]:
                        continue
                    for k in range(j + 1, len(candidates)):
                        vk = cand_values[k]
                        if vi + vj + vk > n:
                            break
                        if vi + vj + vk == n:
                            kk = cand_indices[k]
                            # Check all pairs are incomparable
                            if not divs[ii][kk] and not divs[kk][ii] and not divs[jj][kk] and not divs[kk][jj]:
                                best_answer = [vi, vj, vk]
                                break
                    if best_answer is not None:
                        break
                if best_answer is not None:
                    break
        
        if best_answer is None and time.time() < deadline:
            # Try 4 terms with more systematic search
            found = False
            for i in range(len(candidates)):
                if time.time() > deadline:
                    break
                vi = cand_values[i]
                ii = cand_indices[i]
                for j in range(i + 1, len(candidates)):
                    vj = cand_values[j]
                    if vi + vj >= n:
                        break
                    jj = cand_indices[j]
                    if divs[ii][jj] or divs[jj][ii]:
                        continue
                    for k in range(j + 1, len(candidates)):
                        vk = cand_values[k]
                        if vi + vj + vk >= n:
                            break
                        kk = cand_indices[k]
                        if divs[ii][kk] or divs[kk][ii] or divs[jj][kk] or divs[kk][jj]:
                            continue
                        for l in range(k + 1, len(candidates)):
                            vl = cand_values[l]
                            if vi + vj + vk + vl > n:
                                break
                            if vi + vj + vk + vl == n:
                                ll = cand_indices[l]
                                # Check all pairs
                                ok = True
                                for a_idx, b_idx in [(ii, jj), (ii, kk), (ii, ll), (jj, kk), (jj, ll), (kk, ll)]:
                                    if divs[a_idx][b_idx] or divs[b_idx][a_idx]:
                                        ok = False
                                        break
                                if ok:
                                    best_answer = [vi, vj, vk, vl]
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
        
        if best_answer is None:
            # Fallback: use a valid representation (may not be minimal)
            # Use the seed approach for this target
            for count in range(1, 31):
                if time.time() > deadline:
                    break
                for terms in combinations([x for x in smooth if x <= n], count):
                    if sum(terms) == n:
                        # Check antichain
                        valid = True
                        for a, b in combinations(terms, 2):
                            if b % a == 0:
                                valid = False
                                break
                        if valid:
                            best_answer = list(terms)
                            break
                if best_answer is not None:
                    break
        
        answers.append(best_answer)
    
    return answers
# EVOLVE-BLOCK-END
