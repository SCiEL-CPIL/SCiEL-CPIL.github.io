import time
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # 1. Greedy Initialization
    # Sort by weight descending (1, 2, 3...)
    S = set()
    for x in range(1, N + 1):
        # Check if x can be added
        valid = True
        for a in S:
            for b in S:
                if a == b: continue
                # Check if {x, a, b} is a 3-AP
                # Sort x, a, b
                # Or use the 3 conditions
                # 2a = x+b => b = 2a-x
                # 2b = x+a => a = 2b-x
                # 2x = a+b => a+b = 2x
                
                # We just need to check if a,b,x form AP.
                # It's enough to check if x is the 3rd term of an AP with a,b?
                # No, we need to check all permutations.
                # But since we add x last, we just check if x completes an AP with any pair in S.
                # This is what the seed did.
                # However, for local search, we need to check conflicts properly.
                pass
        
        # The seed logic was:
        # if all(2*b-a!=x for a in selected for b in selected if a<b):
        # This works because x is the largest.
        # In local search, x might be smaller than some elements in S.
        # So we need a robust check.
        
        # Let's use a helper to check if a set is AP-free.
        # But for greedy, x is largest, so seed logic is fine.
        if all(2*b - a != x for a in S for b in S if a < b):
            S.add(x)
            
    best_S = set(S)
    best_score = sum(1.0/i for i in S)
    
    # 2. Local Search
    # We will try to improve S.
    # Move: Add x, remove C(x, S).
    
    def get_conflicts(x, S):
        """Returns set of y in S that conflict with x (i.e., {x,y,z} is AP for some z in S)."""
        conflicts = set()
        for y in S:
            if y == x: continue
            # Check if there exists z in S such that {x,y,z} is AP
            # z1 = 2y - x
            z1 = 2*y - x
            if z1 in S and z1 != y and z1 != x:
                conflicts.add(y)
            # z2 = 2x - y
            z2 = 2*x - y
            if z2 in S and z2 != y and z2 != x:
                conflicts.add(y)
            # z3 = (x+y)/2
            if (x + y) % 2 == 0:
                z3 = (x + y) // 2
                if z3 in S and z3 != y and z3 != x:
                    conflicts.add(y)
        return conflicts

    def score(S):
        return sum(1.0/i for i in S)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_gain = 0
        best_move = None
        
        candidates = [x for x in range(1, N+1) if x not in S]
        
        for x in candidates:
            conflicts = get_conflicts(x, S)
            if not conflicts:
                gain = 1.0/x
            else:
                gain = 1.0/x - sum(1.0/y for y in conflicts)
            
            if gain > best_gain:
                best_gain = gain
                best_move = (x, conflicts)
                
        if best_move:
            x, conflicts = best_move
            S.add(x)
            for y in conflicts:
                S.remove(y)
            improved = True
            
    # Return best found
    return list(best_S)
