"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
def construct():
    import random
    random.seed(42)
    
    def is_sidon(selected):
        sums = set()
        for i in range(len(selected)):
            for j in range(i, len(selected)):
                s = selected[i] + selected[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def is_maximal(selected):
        sums = set()
        for i in range(len(selected)):
            for j in range(i, len(selected)):
                sums.add(selected[i] + selected[j])
        for x in range(1, N + 1):
            if x in selected:
                continue
            # Check if x can be added
            can_add = True
            for a in selected:
                if x + a in sums:
                    can_add = False
                    break
            if can_add and 2 * x in sums:
                can_add = False
            if can_add:
                return False
        return True
    
    # Try different constructions
    # Construction 1: Powers of 2 plus adjustments
    # Construction 2: Random greedy with minimization
    
    best = None
    for seed in range(100):
        random.seed(seed)
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        selected = []
        sums = set()
        for x in candidates:
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        if best is None or len(selected) < len(best):
            best = selected
            if is_maximal(selected):
                break
    
    if best is None or not is_maximal(best):
        # Fallback: use greedy ascending
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        best = selected
    
    return best
# EVOLVE-BLOCK-END
