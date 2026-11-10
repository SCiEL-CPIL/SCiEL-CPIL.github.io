import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    if not (q == 3 and n == 8 and radius == 1):
        # Fallback
        N = q ** n
        if N > 100000:
            return [[i % q for i in range(n)] for i in range(N)]
        
        powers = [q ** i for i in range(n)]
        neighbors = []
        for w in range(N):
            ball = {w}
            digits = [0] * n
            temp = w
            for i in range(n):
                digits[i] = temp % q
                temp //= q
            for pos in range(n):
                original = digits[pos]
                for val in range(q):
                    if val == original:
                        continue
                    digits[pos] = val
                    new_w = 0
                    mult = 1
                    for i in range(n):
                        new_w += digits[i] * mult
                        mult *= q
                    ball.add(new_w)
                digits[pos] = original
            neighbors.append(list(ball))
        
        uncovered = set(range(N))
        code = []
        while uncovered and time.time() < deadline:
            best_cand = -1
            best_count = -1
            for c in uncovered:
                count = sum(1 for nb in neighbors[c] if nb in uncovered)
                if count > best_count:
                    best_count = count
                    best_cand = c
            if best_cand == -1:
                break
            code.append(best_cand)
            for nb in neighbors[best_cand]:
                if nb in uncovered:
                    uncovered.remove(nb)
        if uncovered:
            code.extend(uncovered)
            
        result = []
        for w in code:
            digits = []
            temp = w
            for _ in range(n):
                digits.append(temp % q)
                temp //= q
            result.append(digits)
        return result

    # Specific q=3, n=8, radius=1
    N = q ** n
    powers = [q ** i for i in range(n)]
    neigh_size = 1 + n * (q - 1)
    neighbors = np.zeros((N, neigh_size), dtype=np.int32)
    
    for w in range(N):
        neighbors[w, 0] = w
        idx = 1
        for i in range(n):
            d = (w // powers[i]) % q
            for delta in range(1, q):
                new_d = (d + delta) % q
                diff = new_d - d
                new_w = w + diff * powers[i]
                neighbors[w, idx] = new_w
                idx += 1

    def greedy_init(rng):
        uncovered = np.ones(N, dtype=np.bool_)
        code = set()
        while np.any(uncovered) and time.time() < deadline - 5:
            gains = uncovered[neighbors].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            w = rng.choice(candidates)
            code.add(w)
            uncovered[neighbors[w, :]] = False
        return code

    def local_search(code_set, rng, deadline):
        code_list = list(code_set)
        in_code = np.zeros(N, dtype=np.bool_)
        for c in code_list:
            in_code[c] = True
            
        coverage = np.zeros(N, dtype=np.int32)
        for c in code_list:
            coverage[neighbors[c, :]] += 1
            
        current_size = len(code_list)
        no_improvement_count = 0
        
        while time.time() < deadline - 2:
            improved = False
            
            # 1. Deletion Pass
            # Find redundant codewords
            redundant = []
            for c in code_list:
                if np.min(coverage[neighbors[c, :]]) > 1:
                    redundant.append(c)
                    
            if redundant:
                c = redundant[0]
                # Remove c
                code_list.remove(c)
                in_code[c] = False
                coverage[neighbors[c, :]] -= 1
                current_size -= 1
                improved = True
                no_improvement_count = 0
                continue
            
            # 2. Swap Pass
            # Find codeword with most essential words
            # Essential words for c are those w in N(c) with coverage[w] == 1
            best_c = -1
            max_essential = 0
            best_essential_set = set()
            
            # Sample a subset of codewords to speed up if code is large
            # But 486 is small, so iterate all
            for c in code_list:
                c_neighbors = neighbors[c, :]
                ess = c_neighbors[coverage[c_neighbors] == 1]
                if len(ess) > max_essential:
                    max_essential = len(ess)
                    best_c = c
                    best_essential_set = set(ess.tolist())
            
            if best_c == -1 or max_essential == 0:
                # No essential words? Then all codewords are redundant?
                # If max_essential == 0, it means for every c, min(coverage[N(c)]) > 1?
                # Wait, if min > 1, it would have been caught in Deletion Pass.
                # So if we are here, every c has at least one essential word.
                # If max_essential == 0, it means no codewords? Impossible.
                # So this branch is unlikely unless code_list is empty.
                break

            # Try to find d not in code such that best_essential_set is subset of N(d)
            # d must be in intersection of N(w) for w in best_essential_set
            candidates = set(neighbors[list(best_essential_set)[0], :].tolist())
            for w in list(best_essential_set)[1:]:
                candidates.intersection_update(neighbors[w, :].tolist())
            
            # Filter out existing codewords
            valid_candidates = [d for d in candidates if not in_code[d]]
            
            if valid_candidates:
                d = rng.choice(valid_candidates)
                # Perform swap
                # Update coverage
                # Remove best_c, add d
                coverage[neighbors[best_c, :]] -= 1
                coverage[neighbors[d, :]] += 1
                in_code[best_c] = False
                in_code[d] = True
                code_list.remove(best_c)
                code_list.append(d)
                improved = True
                no_improvement_count = 0
            else:
                # No valid swap for this best_c
                # Try random swaps?
                # Pick random c, random d
                if no_improvement_count > 50:
                    # Perturb
                    if code_list:
                        c = rng.choice(code_list)
                        # Find any d that covers essential words of c
                        c_neighbors = neighbors[c, :]
                        ess = c_neighbors[coverage[c_neighbors] == 1]
                        if len(ess) > 0:
                             candidates = set(neighbors[ess[0], :].tolist())
                             for w in ess[1:]:
                                 candidates.intersection_update(neighbors[w, :].tolist())
                             valid_candidates = [d for d in candidates if not in_code[d]]
                             if valid_candidates:
                                 d = rng.choice(valid_candidates)
                                 coverage[neighbors[c, :]] -= 1
                                 coverage[neighbors[d, :]] += 1
                                 in_code[c] = False
                                 in_code[d] = True
                                 code_list.remove(c)
                                 code_list.append(d)
                                 improved = True
                                 no_improvement_count = 0
                    if not improved:
                         no_improvement_count += 1
                         if no_improvement_count > 100:
                             break
                else:
                    no_improvement_count += 1
                    if no_improvement_count > 10:
                        # Shuffle code_list order to avoid bias?
                        rng.shuffle(code_list)
                        no_improvement_count = 0 # Reset counter after shuffle?

        return set(code_list)

    best_code = None
    best_size = float('inf')
    
    for i in range(50):
        if time.time() > deadline - 5:
            break
            
        rng = np.random.default_rng(42 + i)
        code = greedy_init(rng)
        code = local_search(code, rng, deadline)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
        if best_size <= 486:
            break

    if best_code is None:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        best_code = set()
        for c1 in tetracode:
            for c2 in tetracode:
                best_code.add(int("".join(map(str, c1 + c2)), 3))

    result = []
    for w in best_code:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        result.append(digits)
    
    return result
