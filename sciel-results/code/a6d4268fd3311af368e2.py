import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    target_diffs = set(range(1, 31))
    
    def is_valid_solution(elements):
        diffs = set()
        n = len(elements)
        for i in range(n):
            for j in range(i + 1, n):
                d = elements[j] - elements[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return target_diffs.issubset(diffs)

    def search(D, max_k=32):
        found = [None]
        
        def dfs(elements, diff_set, last, depth):
            if time.time() > deadline:
                return True
            if found[0] is not None:
                return True

            if target_diffs.issubset(diff_set):
                found[0] = list(elements)
                return True

            if depth >= max_k:
                return False
                
            missing = target_diffs - diff_set
            if not missing:
                pass
            
            k_new = max_k - depth
            if k_new <= 0:
                return False
            max_new_diffs = k_new * len(elements) + k_new * (k_new - 1) // 2
            if max_new_diffs < len(missing):
                return False

            max_missing = max(missing) if missing else 0
            if max_missing > D:
                return False

            candidates = []
            for x in range(last + 1, D + 1):
                new_diffs = set()
                valid = True
                for e in elements:
                    d = x - e
                    if d in diff_set:
                        valid = False
                        break
                    new_diffs.add(d)
                if not valid:
                    continue
                covered = len(new_diffs & missing)
                candidates.append((covered, x, new_diffs))
            
            candidates.sort(key=lambda t: -t[0])
            
            for covered, x, new_diffs in candidates:
                if time.time() > deadline:
                    return True
                if found[0] is not None:
                    return True
                
                elements.append(x)
                new_diff_set = diff_set | new_diffs
                if dfs(elements, new_diff_set, x, depth + 1):
                    return True
                elements.pop()
                
            return False

        dfs([0], set(), 0, 1)
        return found[0]

    for D in range(30, 100):
        if time.time() > deadline:
            break
        result = search(D)
        if result is not None:
            if is_valid_solution(result):
                return result

    def greedy_fallback():
        a = [0]
        covered = set()
        while not target_diffs.issubset(covered) and len(a) < 32:
            missing = target_diffs - covered
            best_x = None
            best_score = -1
            best_new_diffs = None
            
            for x in range(max(a) + 1, max(a) + 200):
                new_diffs = set()
                valid = True
                for e in a:
                    d = x - e
                    if d in covered:
                        valid = False
                        break
                    new_diffs.add(d)
                if not valid:
                    continue
                score = len(new_diffs & missing)
                if score > best_score:
                    best_score = score
                    best_x = x
                    best_new_diffs = new_diffs
            
            if best_x is None:
                best_x = max(a) + 100
                while True:
                    new_diffs = set(best_x - e for e in a)
                    if not (new_diffs & covered):
                        break
                    best_x += 1
                best_new_diffs = new_diffs
            
            a.append(best_x)
            covered |= best_new_diffs
            
        return a

    return greedy_fallback()
