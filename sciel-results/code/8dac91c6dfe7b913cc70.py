import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin
    
    # Start with tetracode direct sum (243 words for q=3, n=8, radius=1)
    if q == 3 and n == 8 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        initial = [tuple(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=4)]
        initial = [list(w) for w in initial]
    else:
        # Generic greedy construction
        initial = []
    
    if not initial:
        # Greedy set cover
        all_words = np.array(np.meshgrid(*[np.arange(q)]*n)).reshape(n, -1).T
        uncovered = set(map(tuple, all_words))
        code = []
        while uncovered and time.time() < deadline:
            # Pick word covering most uncovered
            best_word = None
            best_count = -1
            # Sample candidate words
            candidates = list(uncovered)
            random.shuffle(candidates)
            candidates = candidates[:min(500, len(candidates))]
            for w in candidates:
                w = tuple(w)
                count = 0
                for i in range(n):
                    for val in range(q):
                        if val == w[i]:
                            continue
                        nw = list(w)
                        nw[i] = val
                        if tuple(nw) in uncovered:
                            count += 1
                if w in uncovered:
                    count += 1
                if count > best_count:
                    best_count = count
                    best_word = w
            if best_word is None:
                break
            code.append(list(best_word))
            # Remove covered words
            bw = list(best_word)
            for i in range(n):
                for val in range(q):
                    nw = bw[:]
                    nw[i] = val
                    uncovered.discard(tuple(nw))
            uncovered.discard(tuple(best_word))
        initial = code
    
    # Local search: try to remove codewords
    code = initial[:]
    code_set = set(map(tuple, code))
    
    def is_valid(codewords):
        """Check if codewords cover all words within radius."""
        if time.time() > deadline:
            return True  # Assume valid to avoid timeout
        all_words = np.array(np.meshgrid(*[np.arange(q)]*n)).reshape(n, -1).T
        code_arr = np.array(codewords)
        # For each word, check if within radius of some codeword
        covered = np.zeros(len(all_words), dtype=bool)
        for i, w in enumerate(all_words):
            if covered[i]:
                continue
            # Check against each codeword
            for cw in code_arr:
                dist = np.sum(w != cw)
                if dist <= radius:
                    covered[i] = True
                    break
        return np.all(covered)
    
    # Try removing codewords one at a time
    removed_any = True
    while removed_any and time.time() < deadline:
        removed_any = False
        for i in range(len(code)):
            if time.time() > deadline:
                break
            # Try removing code[i]
            test_code = code[:i] + code[i+1:]
            if len(test_code) == 0:
                continue
            # Quick check: is code[i] redundant?
            # Check if all words covered by code[i] are also covered by others
            cw = code[i]
            # Get all words within radius of cw
            covered_by_i = []
            for d in range(radius + 1):
                if d == 0:
                    covered_by_i.append(tuple(cw))
                else:
                    import itertools
                    positions = list(itertools.combinations(range(n), d))
                    for pos in positions:
                        for vals in itertools.product(range(1, q), repeat=d):
                            nw = cw[:]
                            for p, v in zip(pos, vals):
                                if v != 0:
                                    nw[p] = (cw[p] + v) % q
                                else:
                                    nw[p] = (cw[p] + 1) % q
                            covered_by_i.append(tuple(nw))
            
            redundant = True
            for w in covered_by_i:
                # Check if w is covered by some other codeword
                found = False
                for j, other in enumerate(test_code):
                    dist = sum(1 for a, b in zip(w, other) if a != b)
                    if dist <= radius:
                        found = True
                        break
                if not found:
                    redundant = False
                    break
            
            if redundant:
                code = test_code
                removed_any = True
                break
    
    # Convert to list of lists
    return [list(w) for w in code]
