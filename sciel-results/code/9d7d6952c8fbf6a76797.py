import time
import random

def construct():
    start_time = time.time()
    time_limit = 58.0  # Safety margin for 60s limit
    
    n = 256
    candidates = list(range(1, n + 1))
    
    best_size = 0
    best_A = []
    best_B = []
    
    def is_sidon(S):
        if not S:
            return True
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def get_diffs(S):
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                diffs.add(abs(S[i] - S[j]))
        return diffs

    def get_sums(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        return sums

    def try_build_pair(order, max_A_size=16, max_B_size=16):
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        forbidden_diffs = set() # Union of diffs_A and diffs_B for checking new additions
        
        def can_add_to(x, S, sums_S, diffs_S):
            # Check Sidon property for S + {x}
            # New sums: x + a for a in S, and 2x
            if 2 * x in sums_S:
                return False
            for a in S:
                if x + a in sums_S:
                    return False
            # Check differences for disjointness
            # New diffs: |x - a| for a in S
            for a in S:
                d = abs(x - a)
                if d in forbidden_diffs:
                    return False
            # Also, the new diffs must not conflict with each other?
            # No, diffs within the same set are already handled by Sidon?
            # Wait. Sidon property is about SUMS.
            # Difference disjointness is between A and B.
            # Differences within A can be anything as long as A is Sidon.
            # BUT, if we add x to A, the new diffs |x-a| become part of D_A.
            # These new diffs must not be in D_B.
            # They also must not be in D_A? No, D_A is the set of diffs of A.
            # If A is Sidon, are diffs unique? No.
            # Example: {1, 2, 4}. Diffs: 1, 2, 3. Unique.
            # Example: {1, 3, 4}. Sums: 2, 4, 5, 6, 7, 8. Sidon. Diffs: 2, 1, 3.
            # Can a Sidon set have duplicate differences?
            # If |a-b| = |c-d| = d, and {a,b} != {c,d}.
            # This is allowed in Sidon sets.
            # So we just need to ensure new diffs are not in D_B.
            
            return True

        def add_to(x, S, sums_S, diffs_S):
            sums_S.add(2 * x)
            for a in S:
                sums_S.add(x + a)
            
            new_diffs = set()
            for a in S:
                d = abs(x - a)
                new_diffs.add(d)
            
            diffs_S.update(new_diffs)
            forbidden_diffs.update(new_diffs)
            S.append(x)

        for x in order:
            # Try to add to A
            if len(A) < max_A_size:
                if can_add_to(x, A, sums_A, diffs_A):
                    add_to(x, A, sums_A, diffs_A)
                    continue
            
            # Try to add to B
            if len(B) < max_B_size:
                if can_add_to(x, B, sums_B, diffs_B):
                    add_to(x, B, sums_B, diffs_B)
                    continue
            
            # If neither, skip
            
        return len(A) + len(B), A, B

    # Strategy:
    # 1. Try to build A first with a specific order, then B.
    # 2. Randomize the order.
    # 3. Also try to balance the sizes.
    
    # We can also try to construct A deterministically first to a good size,
    # then randomize B construction.
    
    # Let's just do random restarts with a slightly smarter heuristic:
    # Sort candidates by "value" or just shuffle.
    
    restarts = 0
    while time.time() - start_time < time_limit:
        random.shuffle(candidates)
        # Try to fill A and B
        size, A, B = try_build_pair(candidates)
        
        if size > best_size:
            best_size = size
            best_A = A[:]
            best_B = B[:]
            
            # If we hit the theoretical max (32), we can stop or verify
            if best_size >= 32:
                break
        
        restarts += 1
        # If we are stuck at a low size for many restarts, maybe change strategy?
        # For now, just keep restarting.

    if best_size == 0:
        return [[1], [2]]
        
    return [best_A, best_B]
