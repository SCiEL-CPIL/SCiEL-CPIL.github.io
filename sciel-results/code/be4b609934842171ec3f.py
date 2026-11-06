import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    N = 256
    # 1. Generate initial greedy solution
    # Iterate 2..256, add if no conflict with current set
    selected = []
    for x in range(2, N + 1):
        if all(lcm(x, y) > N for y in selected):
            selected.append(x)
    
    # Convert to set for O(1) lookup
    selected_set = set(selected)
    
    # Precompute conflict matrix or list for speed?
    # N=256, N^2 = 65536. lcm is fast.
    # conflicts[x] = set of y such that lcm(x,y) <= 256
    # We only care about x in 2..256.
    # Let's compute conflicts for all pairs once.
    # conflicts[i] is a bitset or set of j.
    
    # Optimization: 
    # We need to check validity of (S - {u}) U {v}.
    # This is valid if v is not in S (true) and v has no conflicts with (S - {u}).
    # i.e., for all w in S, w != u, lcm(v, w) > 256.
    
    # Let's build conflict sets for all numbers 2..256
    # conflict_set[x] = { y in 2..256 | lcm(x,y) <= 256 }
    conflict_sets = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        for j in range(i + 1, N + 1):
            if lcm(i, j) <= N:
                conflict_sets[i].add(j)
                conflict_sets[j].add(i)
    
    # Local Search
    # Keep a list of selected numbers
    # We want to maximize sum(1/x).
    # Current sum
    current_sum = sum(1.0 / x for x in selected)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all swaps (u in S, v not in S)
        # To speed up, iterate u in S, then v in candidates.
        # Candidates are those v not in S that might be beneficial (1/v > 1/u => v < u).
        # But v must be compatible with S \ {u}.
        
        best_swap = None
        best_gain = 0
        
        for u in selected:
            # If we remove u, which v's become valid?
            # v is valid if conflict_sets[v] & (selected_set - {u}) is empty.
            # This means all conflicts of v are in {u} (or empty).
            # So v can only conflict with u among the current set.
            
            # Iterate v in 2..256 not in selected_set
            # Optimization: only consider v < u?
            # If v > u, 1/v < 1/u, so swap decreases sum. 
            # UNLESS removing u allows adding MULTIPLE v's? 
            # We are doing single swap here. So v must be < u to improve.
            for v in range(2, u): 
                if v in selected_set:
                    continue
                
                # Check if v is compatible with S \ {u}
                # v's conflicts in S must be subset of {u}
                # i.e., conflict_sets[v] & selected_set == {u} or empty?
                # If empty, v could have been added already (but it wasn't, so it must conflict with someone).
                # Since S is maximal, v must conflict with at least one element in S.
                # If we remove u, v is valid if all its conflicts in S are exactly u (or subset of {u}).
                
                # Efficient check:
                # We can precompute for each v, the list of conflicts in S.
                # But S changes.
                
                # Simple check:
                # Is there any w in S, w != u, such that w in conflict_sets[v]?
                # If yes, invalid.
                
                valid = True
                for w in selected:
                    if w == u:
                        continue
                    if w in conflict_sets[v]:
                        valid = False
                        break
                
                if valid:
                    gain = (1.0 / v) - (1.0 / u)
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (u, v)
        
        if best_swap:
            u, v = best_swap
            selected.remove(u)
            selected_set.remove(u)
            selected.append(v)
            selected_set.add(v)
            current_sum += best_gain
            improved = True
    
    # Optional: Try 2-for-1 moves if time permits
    # Remove u, add v, w.
    # 1/v + 1/w > 1/u.
    # v, w must be compatible with S\{u} and each other.
    
    if time.time() < deadline - 5:
        improved2 = True
        while improved2 and time.time() < deadline - 2:
            improved2 = False
            # Heuristic: only try removing u if it's large (small weight) 
            # and try adding small v, w.
            # This is expensive. O(N^3).
            # N=256. 256^3 = 16M. Might be too slow in Python.
            # Skip 2-for-1 if single swap didn't help much or if time is tight.
            pass

    return selected

if __name__ == "__main__":
    s = construct()
    print(len(s))
    print(sum(1/x for x in s))
