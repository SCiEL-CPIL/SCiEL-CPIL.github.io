import time
import random

def construct():
    # Deadline: 90s limit, use 85s for safety
    deadline = time.time() + 85.0
    N = 128
    
    # 1. Pre-compute forbidden hyperedges
    # A violation is a set of values V = {a, b, c, a+b, a+c, b+c} 
    # where a < b < c are distinct, and all values in V are in 1..N.
    # We store unique frozensets to handle cases where some sums equal elements.
    edges = []
    seen_edges = set()
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            # We need c > b such that a+c <= N and b+c <= N
            max_c = min(N - a, N - b)
            if max_c <= b:
                continue
            for c in range(b + 1, max_c + 1):
                s1 = a + b
                s2 = a + c
                s3 = b + c
                # All sums are <= N by construction of max_c
                # Check if sums are distinct from a,b,c? 
                # The problem says "three DISTINCT members a,b,c". 
                # It does NOT say the sums must be distinct from a,b,c or each other.
                # However, the condition is: a,b,c in S AND a+b, a+c, b+c in S.
                # So the "bad" set is the set of all values involved.
                v = {a, b, c, s1, s2, s3}
                # All values are >= 1 and <= N
                if v not in seen_edges:
                    seen_edges.add(v)
                    edges.append(frozenset(v))
    
    # 2. Index edges by element for fast conflict checking
    # elem_to_edges[x] = list of indices of edges containing x
    elem_to_edges = {i: [] for i in range(1, N + 1)}
    for i, e in enumerate(edges):
        for v in e:
            elem_to_edges[v].append(i)

    def check_conflicts(S_set, x):
        """
        Returns True if adding x to S_set creates a violation.
        Assumes x is not in S_set.
        A violation is created if there is an edge e containing x 
        such that all OTHER elements in e are already in S_set.
        """
        for idx in elem_to_edges[x]:
            e = edges[idx]
            # Check if e - {x} is subset of S_set
            # e contains x.
            # We need to check if every other element in e is in S_set.
            # Since e is small (size <= 6), this is fast.
            # Optimization: if x is in e, we check the rest.
            # Note: e is a frozenset.
            if e.difference({x}).issubset(S_set):
                return True
        return False

    def is_valid_swap(S_set, x_out, x_in):
        """
        Check if S_set - {x_out} + {x_in} is valid.
        x_out is in S_set, x_in is not in S_set.
        Violations can only be created involving x_in.
        So we check if adding x_in to (S_set - {x_out}) creates a conflict.
        """
        # Temporarily remove x_out
        S_new = S_set - {x_out}
        return not check_conflicts(S_new, x_in)

    def solve():
        # Initialize S with upper half, which is always valid
        # Smallest sum of two distinct elements in {65..128} is 65+66=131 > 128
        # So no sum of any two elements in S can be in S.
        S = set(range(N // 2 + 1, N + 1))
        best_S = set(S)
        best_size = len(S)
        
        # Candidates to add: lower half
        candidates = list(range(1, N // 2 + 1))
        random.shuffle(candidates)
        
        # Phase 1: Greedy Addition
        # Try to add as many candidates as possible
        while True:
            added_any = False
            # Shuffle to break symmetry/randomness
            random.shuffle(candidates)
            for x in candidates:
                if x in S:
                    continue
                if not check_conflicts(S, x):
                    S.add(x)
                    added_any = True
            if not added_any:
                break
        
        if len(S) > best_size:
            best_S = set(S)
            best_size = len(S)
        
        # Phase 2: Local Search with Swaps
        # Try to swap elements to enable more additions
        # We will iterate while time permits.
        
        # Create a list of all numbers not in S
        not_in_S = [i for i in range(1, N + 1) if i not in S]
        
        while time.time() < deadline - 2.0:
            # Randomly pick an element to remove and one to add
            if not not_in_S:
                break
            
            # Pick x_out from S
            # We can pick randomly, or pick one that is "least useful" (harder to optimize)
            # For now, random pick.
            x_out = random.choice(list(S))
            x_in = random.choice(not_in_S)
            
            # Check if swap is valid
            if is_valid_swap(S, x_out, x_in):
                # Perform swap
                S.remove(x_out)
                S.add(x_in)
                
                # Update not_in_S
                # Remove x_in from not_in_S, add x_out
                # This is O(N) but N=128, so it's fine.
                # Or we can just rebuild it occasionally.
                # Let's just check membership in the loop.
                
                # After swap, try to add more elements
                # This is crucial: a swap might "unlock" several additions.
                added_any = True
                while added_any:
                    added_any = False
                    # Try to add from the current not_in_S list
                    # We need a fresh list or just iterate through all numbers not in S
                    for y in range(1, N + 1):
                        if y in S:
                            continue
                        if not check_conflicts(S, y):
                            S.add(y)
                            added_any = True
                
                if len(S) > best_size:
                    best_S = set(S)
                    best_size = len(S)
            
            # Occasionally, if we are stuck, we might want to do a "kick"
            # e.g., remove 2 elements and add 2 different ones?
            # Or just rely on random swaps.
            
            # To prevent getting stuck in a small cycle, we can occasionally
            # revert to best_S and restart the greedy phase?
            # Or just keep swapping.
            
            # A simple improvement: if we haven't improved in a long time,
            # maybe reset to best_S and try a different random seed/order?
            # Given the time budget, random swaps are usually sufficient for N=128.
            
            # Let's add a small probability of resetting to best_S to explore new areas
            if random.random() < 0.01:
                S = set(best_S)
                
    solve()
    
    result = sorted(best_S)
    return list(result)

if __name__ == "__main__":
    # Quick self-test
    s = construct()
    print(f"Size: {len(s)}")
    # Verify
    S = set(s)
    valid = True
    for a in S:
        for b in S:
            if a >= b: continue
            for c in S:
                if b >= c: continue
                if a+b in S and a+c in S and b+c in S:
                    valid = False
                    print(f"Violation: {a},{b},{c} -> {a+b},{a+c},{b+c}")
    print(f"Valid: {valid}")
