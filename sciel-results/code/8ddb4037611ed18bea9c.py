import time
import random

def construct():
    start_time = time.time()
    deadline = 88.0  # Safety margin for 90s limit
    n = 256
    
    def elapsed():
        return time.time() - start_time

    def is_valid_solution(A, B):
        if not A or not B:
            return False
        if len(A) > 64 or len(B) > 64:
            return False
        
        # Check Sidon for A
        sums_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums_A:
                    return False
                sums_A.add(s)
        
        # Check Sidon for B
        sums_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                s = B[i] + B[j]
                if s in sums_B:
                    return False
                sums_B.add(s)
        
        # Check disjoint differences
        diffs_A = set()
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                diffs_A.add(abs(A[j] - A[i]))
        diffs_B = set()
        for i in range(len(B)):
            for j in range(i+1, len(B)):
                diffs_B.add(abs(B[j] - B[i]))
                
        if diffs_A & diffs_B:
            return False
            
        return True

    def can_add_to_A(x, A, sums_A, diffs_B):
        """Check if x can be added to A without violating Sidon or Diff Disjointness."""
        # Sidon check
        if 2 * x in sums_A:
            return False
        for a in A:
            if (x + a) in sums_A:
                return False
        
        # Diff disjointness check
        # New diffs |x - a| must not be in diffs_B
        for a in A:
            if abs(x - a) in diffs_B:
                return False
        return True

    def can_add_to_B(x, B, sums_B, diffs_A):
        """Check if x can be added to B without violating Sidon or Diff Disjointness."""
        # Sidon check
        if 2 * x in sums_B:
            return False
        for b in B:
            if (x + b) in sums_B:
                return False
        
        # Diff disjointness check
        for b in B:
            if abs(x - b) in diffs_A:
                return False
        return True

    def add_to_A(x, A, sums_A, diffs_A):
        A.append(x)
        sums_A.add(2 * x)
        for a in A[:-1]:
            sums_A.add(x + a)
            diffs_A.add(abs(x - a))

    def add_to_B(x, B, sums_B, diffs_B):
        B.append(x)
        sums_B.add(2 * x)
        for b in B[:-1]:
            sums_B.add(x + b)
            diffs_B.add(abs(x - b))

    def get_state(A, B):
        sums_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums_A.add(A[i] + A[j])
        sums_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                sums_B.add(B[i] + B[j])
        diffs_A = set()
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                diffs_A.add(abs(A[i] - A[j]))
        diffs_B = set()
        for i in range(len(B)):
            for j in range(i+1, len(B)):
                diffs_B.add(abs(B[i] - B[j]))
        return sums_A, sums_B, diffs_A, diffs_B

    def try_insert(A, B, unused, sums_A, sums_B, diffs_A, diffs_B):
        """Try to insert any element from unused into A or B."""
        if not unused:
            return False
        
        # Shuffle to randomize order
        local_unused = list(unused)
        random.shuffle(local_unused)
        
        for x in local_unused:
            if can_add_to_A(x, A, sums_A, diffs_B):
                add_to_A(x, A, sums_A, diffs_A)
                unused.remove(x)
                return True
            if can_add_to_B(x, B, sums_B, diffs_A):
                add_to_B(x, B, sums_B, diffs_B)
                unused.remove(x)
                return True
        return False

    def greedy_build():
        A = []
        B = []
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        sums_A, sums_B, diffs_A, diffs_B = set(), set(), set(), set()
        
        # Greedy pass: try to add each candidate
        for x in candidates:
            if can_add_to_A(x, A, sums_A, diffs_B):
                add_to_A(x, A, sums_A, diffs_A)
            elif can_add_to_B(x, B, sums_B, diffs_A):
                add_to_B(x, B, sums_B, diffs_B)
                
        # Rebuild state to be sure
        sums_A, sums_B, diffs_A, diffs_B = get_state(A, B)
        
        # Calculate unused
        used = set(A) | set(B)
        unused = [x for x in range(1, n + 1) if x not in used]
        
        return A, B, unused, sums_A, sums_B, diffs_A, diffs_B

    def local_search(A, B, unused, sums_A, sums_B, diffs_A, diffs_B):
        # Try to insert more
        # We limit steps to avoid getting stuck in a long loop
        for _ in range(500): 
            if elapsed() > deadline:
                break
            
            if not try_insert(A, B, unused, sums_A, sums_B, diffs_A, diffs_B):
                # Stuck. Kick: remove a random element from the larger set (or random)
                if random.random() < 0.5 and len(A) > 0:
                    x = random.choice(A)
                    A.remove(x)
                    unused.append(x)
                elif len(B) > 0:
                    x = random.choice(B)
                    B.remove(x)
                    unused.append(x)
                else:
                    break
                
                # Rebuild state after removal
                sums_A, sums_B, diffs_A, diffs_B = get_state(A, B)
                
        return A, B, unused

    best_score = 0
    best_sets = ([1], [2])
    
    # Run multiple iterations of Greedy + Local Search
    while elapsed() < deadline:
        A, B, unused, sums_A, sums_B, diffs_A, diffs_B = greedy_build()
        
        # Local search to polish
        A, B, unused = local_search(A, B, unused, sums_A, sums_B, diffs_A, diffs_B)
        
        # Validate final result
        if is_valid_solution(A, B):
            score = len(A) + len(B)
            if score > best_score:
                best_score = score
                best_sets = (A[:], B[:])
                
    return list(best_sets)
