import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    def is_valid(s):
        prods = {1}
        for x in s:
            new_prods = set()
            for p in prods:
                np = x * p
                if np in prods:
                    return False
                new_prods.add(np)
            prods = new_prods
        return True

    # Base valid 16-element set
    base = [2, 3, 4, 5, 7, 9, 11, 13, 16, 17, 19, 23, 25, 29, 31, 37]
    
    if not is_valid(base):
        base = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    best = list(base)
    
    # Try greedy extension with backtracking for 1 more element
    in_best = set(best)
    for x in range(2, 41):
        if time.time() > deadline - 5:
            break
        if x in in_best:
            continue
        candidate = best + [x]
        if is_valid(candidate):
            best = candidate
            in_best.add(x)
            # Try to add one more
            for y in range(2, 41):
                if time.time() > deadline - 5:
                    break
                if y in in_best:
                    continue
                candidate2 = best + [y]
                if is_valid(candidate2):
                    best = candidate2
                    break
    
    return [int(x) for x in best]

if __name__ == "__main__":
    res = construct()
    print(f"Constructed set size: {len(res)}")
    print(res)
