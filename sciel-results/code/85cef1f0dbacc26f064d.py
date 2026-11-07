import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    # Start with the known 16-element construction
    M = 17
    base_set = [1 + M * i for i in range(16)]
    
    # Try to greedily add more elements
    # Track sums by cardinality
    def get_sums_by_card(arr):
        sums = {}
        for k in range(len(arr) + 1):
            s = set()
            for combo in combinations(arr, k):
                s.add(sum(combo))
            sums[k] = s
        return sums
    
    # For the base set, compute sums
    current = list(base_set)
    sums_by_card = get_sums_by_card(current)
    
    # Candidate elements: 1..256 not in current
    candidates = [x for x in range(1, 257) if x not in set(current)]
    
    # Greedily try to add candidates
    added = True
    while added and time.time() < deadline:
        added = False
        for c in candidates[:]:
            if time.time() > deadline:
                break
            new_arr = current + [c]
            # Check if adding c creates cross-cardinality collision
            # New sums: for each existing sum of size k, we get sum+c of size k+1
            # Also the element c itself (size 1)
            ok = True
            for k in range(len(current) + 1):
                new_sums = {s + c for s in sums_by_card[k]}
                new_size = k + 1
                if new_size not in sums_by_card:
                    sums_by_card[new_size] = set()
                # Check collision with existing sums of different cardinality
                for kk in sums_by_card:
                    if kk != new_size:
                        if new_sums & sums_by_card[kk]:
                            ok = False
                            break
                if not ok:
                    break
            if ok:
                # Also check new_sums against each other for same cardinality (allowed)
                # Update sums_by_card
                for k in range(len(current) + 1, -1, -1):
                    if k + 1 not in sums_by_card:
                        sums_by_card[k + 1] = set()
                    sums_by_card[k + 1] |= {s + c for s in sums_by_card[k]}
                # The element c is size 1
                if 1 not in sums_by_card:
                    sums_by_card[1] = set()
                sums_by_card[1].add(c)
                current.append(c)
                candidates.remove(c)
                added = True
                break
    
    return list(current)
