import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    MAX_VAL = 128
    MAX_SUM = 256
    
    # Precompute coprime table
    coprime = [[True] * (MAX_SUM + 1) for _ in range(MAX_SUM + 1)]
    for i in range(1, MAX_SUM + 1):
        for j in range(i, MAX_SUM + 1):
            if math.gcd(i, j) != 1:
                coprime[i][j] = False
                coprime[j][i] = False
    
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = 5
    
    def search_for_B(A, max_val, current_best):
        a1, a2, a3 = A[0], A[1], A[2]
        best_B = None
        best_s = current_best
        
        valid_b_cols = []
        for b in range(1, max_val + 1):
            s1 = a1 + b
            s2 = a2 + b
            s3 = a3 + b
            if coprime[s1][s2] and coprime[s1][s3] and coprime[s2][s3]:
                valid_b_cols.append((b, (s1, s2, s3)))
        
        n_valid = len(valid_b_cols)
        if n_valid < 3:
            return None, current_best
        
        if n_valid > 50:
            valid_b_cols = valid_b_cols[:50]
            n_valid = 50
        
        for i in range(n_valid):
            b1, col1 = valid_b_cols[i]
            for j in range(i + 1, n_valid):
                b2, col2 = valid_b_cols[j]
                ok = True
                for ns in col2:
                    for ps in col1:
                        if not coprime[ns][ps]:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                col12 = col1 + col2
                for k in range(j + 1, n_valid):
                    b3, col3 = valid_b_cols[k]
                    ok = True
                    for ns in col3:
                        for ps in col12:
                            if not coprime[ns][ps]:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue
                    all_sums = col12 + col3
                    score = len(set(all_sums))
                    if score > best_s:
                        best_s = score
                        best_B = [b1, b2, b3]
                        if best_s >= 9:
                            return best_B, best_s
        return best_B, best_s
    
    # Generate candidate A values sorted by potential
    candidate_As = []
    for a1 in range(1, 45):
        for a2 in range(a1 + 1, min(a1 + 20, 45)):
            for a3 in range(a2 + 1, min(a2 + 15, 45)):
                A = [a1, a2, a3]
                cvb = 0
                for b in range(1, 129):
                    s1 = a1 + b
                    s2 = a2 + b
                    s3 = a3 + b
                    if coprime[s1][s2] and coprime[s1][s3] and coprime[s2][s3]:
                        cvb += 1
                if cvb >= 4:
                    candidate_As.append((cvb, A))
    
    candidate_As.sort(key=lambda x: -x[0])
    
    for cvb, A in candidate_As[:300]:
        if time.time() > deadline:
            break
        best_B, score = search_for_B(A, 128, best_score)
        if score > best_score:
            best_score = score
            best_A = A
            best_B = best_B
            if best_score >= 9:
                return [list(best_A), list(best_B)]
    
    # Fallback: broader search
    if best_score < 9:
        for a1 in range(1, 55):
            if time.time() > deadline:
                break
            for a2 in range(a1 + 1, 55):
                if time.time() > deadline:
                    break
                for a3 in range(a2 + 1, 55):
                    if time.time() > deadline:
                        break
                    A = [a1, a2, a3]
                    best_B, score = search_for_B(A, 128, best_score)
                    if score > best_score:
                        best_score = score
                        best_A = A
                        best_B = best_B
                        if best_score >= 9:
                            return [list(best_A), list(best_B)]
    
    return [list(best_A), list(best_B)]
