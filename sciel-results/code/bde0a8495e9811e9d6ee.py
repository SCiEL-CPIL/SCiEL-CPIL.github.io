import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    points11 = list(range(11))
    all_3_subsets = set(combinations(points11, 3))
    candidates11 = list(combinations(points11, 5))
    
    # Map 3-subset to candidate indices
    sub3_to_cands = {s: [] for s in all_3_subsets}
    for i, c in enumerate(candidates11):
        for s in combinations(c, 3):
            sub3_to_cands[s].append(i)
    
    # Symmetry breaking: fix first block
    first_block = (0, 1, 3, 4, 5)
    first_idx = candidates11.index(first_block)
    
    covered3 = set(combinations(first_block, 3))
    blocks11 = [first_block]
    uncovered3 = all_3_subsets - covered3
    
    def backtrack11(covered, current, uncovered):
        if time.time() > deadline:
            return None
        if len(current) == 66:
            return current
        if not uncovered:
            return current if len(current) == 66 else None
        
        # Pick most constrained: fewest valid candidates
        best_target = None
        best_valid = None
        best_count = float('inf')
        
        for target in uncovered:
            valid = []
            for idx in sub3_to_cands[target]:
                c = candidates11[idx]
                if not any(s in covered for s in combinations(c, 3)):
                    valid.append(idx)
            if len(valid) < best_count:
                best_count = len(valid)
                best_target = target
                best_valid = valid
                if best_count == 0:
                    break
                if best_count == 1:
                    break
        
        if best_valid is None or len(best_valid) == 0:
            return None
        
        for idx in best_valid:
            c = candidates11[idx]
            new_cov = set(covered)
            for s in combinations(c, 3):
                new_cov.add(s)
            new_unc = uncovered - new_cov
            res = backtrack11(new_cov, current + [c], new_unc)
            if res is not None:
                return res
        return None
    
    sol11 = backtrack11(covered3, blocks11, uncovered3)
    
    if sol11 is None:
        # Fallback greedy
        covered3 = set()
        blocks11 = []
        for b in candidates11:
            if not any(s in covered3 for s in combinations(b, 3)):
                blocks11.append(b)
                covered3.update(combinations(b, 3))
                if len(blocks11) == 66:
                    break
        sol11 = blocks11
    
    # Double to get S(5,6,12)
    final_blocks = []
    for b in sol11:
        final_blocks.append(list(b) + [11])
        b_set = set(b)
        comp = [p for p in points11 if p not in b_set]
        final_blocks.append(comp)
    
    return final_blocks
