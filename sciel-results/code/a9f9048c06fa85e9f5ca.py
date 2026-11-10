import time
import heapq
import random
import itertools

def construct(q, n, radius):
    # Task is specifically for q=3, n=7, radius=1
    # Fallback for other parameters
    if q != 3 or n != 7 or radius != 1:
        # Simple fallback: direct sum with tetracode if possible, else greedy
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            # This construction is valid for covering radius 1?
            # Tetracode is perfect code of radius 1 in length 4.
            # Direct sum with arbitrary tail: distance to tail is 0, to head is <=1.
            # So yes, it covers radius 1.
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Generic greedy
            # This part is rarely used for this specific task
            pass

    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Space size
    N = 3**n
    
    # Helper: convert integer to tuple of digits
    def int_to_tuple(x):
        t = []
        for _ in range(n):
            t.append(x % 3)
            x //= 3
        return tuple(t)
    
    # Helper: convert tuple to integer
    def tuple_to_int(t):
        x = 0
        for i, d in enumerate(reversed(t)):
            x += d * (3**i)
        return x

    # Precompute neighbors for each word index
    # neighbors[i] is a list of indices j such that dist(i, j) <= 1
    neighbors = [[] for _ in range(N)]
    for i in range(N):
        w = int_to_tuple(i)
        neighbors[i].append(i)
        for pos in range(n):
            orig = w[pos]
            for d in range(3):
                if d == orig:
                    continue
                new_w = list(w)
                new_w[pos] = d
                j = tuple_to_int(tuple(new_w))
                neighbors[i].append(j)
                
    # Best-First Greedy Algorithm
    def run_greedy(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        covered = [False] * N
        # Priority: number of uncovered neighbors
        # Initially, all are uncovered, so priority is len(neighbors[i]) which is 15
        # We use a max-heap, so we push -priority
        # To handle ties and randomization, we add a small random noise or just pick randomly among max
        # For efficiency, we'll use a list of (priority, index) and pick max.
        # But updating priorities is easier with a heap if we use lazy deletion.
        
        # Let's use a simpler approach: 
        # Maintain a list of uncovered indices.
        # At each step, find the index in uncovered with max coverage.
        # Coverage of i = sum(1 for j in neighbors[i] if not covered[j])
        
        # Optimization: 
        # We can maintain an array `coverage_count` for each word, representing how many of its neighbors are currently uncovered.
        # Initially, coverage_count[i] = 15 for all i.
        # When we pick word w and mark it covered:
        #   For each neighbor v of w:
        #     if not covered[v]:
        #       coverage_count[v] -= 1
        
        covered = [False] * N
        coverage_count = [15] * N # 1 self + 14 neighbors
        uncovered_set = set(range(N))
        
        code = []
        
        while uncovered_set:
            # Find the word in uncovered_set with max coverage_count
            # To avoid O(N) scan every time, we could use a heap, but N=2187 is small.
            # O(N * K) where K is number of codewords. 2187 * 200 ~ 437k operations. Very fast.
            
            best_word = None
            max_cov = -1
            # Random tie-breaking
            if random_seed is not None:
                # Iterate and pick max with random tie break
                for w in uncovered_set:
                    c = coverage_count[w]
                    if c > max_cov:
                        max_cov = c
                        best_word = w
                    elif c == max_cov:
                        # Random choice among equals
                        if random.random() < 0.5:
                            best_word = w
            else:
                for w in uncovered_set:
                    if coverage_count[w] > max_cov:
                        max_cov = coverage_count[w]
                        best_word = w
            
            if best_word is None:
                break
                
            code.append(best_word)
            
            # Mark best_word and its neighbors as covered
            for v in neighbors[best_word]:
                if not covered[v]:
                    covered[v] = True
                    uncovered_set.remove(v)
                    # Update coverage_count for neighbors of v
                    # If v is covered, it no longer contributes to the coverage count of its neighbors
                    for u in neighbors[v]:
                        if not covered[u]:
                            coverage_count[u] -= 1
                            
        # Convert indices back to tuples
        return [int_to_tuple(c) for c in code]

    best_code = None
    best_len = float('inf')
    
    # Run multiple iterations with different random seeds to escape local optima
    # The greedy algorithm is deterministic without randomness, so we need randomness in tie-breaking
    # or initial perturbation.
    
    # First, run a deterministic version (or fixed seed)
    code = run_greedy(42)
    if len(code) < best_len:
        best_len = len(code)
        best_code = code
        
    # Then run several random iterations
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        code = run_greedy(i)
        if len(code) < best_len:
            best_len = len(code)
            best_code = code
            
    # Local Search: Try to remove codewords
    # Convert best_code to set of indices for faster lookup
    # We need to verify coverage and try removing.
    
    def verify_and_optimize(code_tuples):
        # Convert to indices
        code_indices = [tuple_to_int(t) for t in code_tuples]
        covered = [False] * N
        
        # Mark covered
        for c in code_indices:
            for v in neighbors[c]:
                covered[v] = True
        
        # Check if all covered
        if not all(covered):
            return code_tuples
            
        # Try to remove each codeword
        # For each codeword c, check if all words in its "exclusive" coverage are covered by others.
        # A word v is exclusively covered by c if dist(v,c)<=1 and for all other c' in code, dist(v,c')>1.
        # This is expensive to compute directly.
        
        # Simpler: For each c in code, temporarily remove it, check if any word becomes uncovered.
        # If not, remove it permanently.
        
        # To make this faster, we can count how many codewords cover each word.
        # cover_count[v] = number of c in code such that dist(v,c)<=1
        # If we remove c, any v with cover_count[v]==1 and dist(v,c)<=1 will become uncovered.
        
        cover_count = [0] * N
        for c in code_indices:
            for v in neighbors[c]:
                cover_count[v] += 1
        
        new_code_indices = []
        for c in code_indices:
            # Check if removing c leaves any hole
            safe = True
            for v in neighbors[c]:
                if cover_count[v] == 1:
                    safe = False
                    break
            if safe:
                # Remove c
                for v in neighbors[c]:
                    cover_count[v] -= 1
            else:
                new_code_indices.append(c)
                
        # Repeat until no more removals possible
        changed = True
        while changed:
            changed = False
            new_new_code = []
            for c in new_code_indices:
                safe = True
                for v in neighbors[c]:
                    if cover_count[v] == 1:
                        safe = False
                        break
                if safe:
                    changed = True
                    for v in neighbors[c]:
                        cover_count[v] -= 1
                else:
                    new_new_code.append(c)
            new_code_indices = new_new_code
            
        return [int_to_tuple(c) for c in new_code_indices]

    optimized_code = verify_and_optimize(best_code)
    if len(optimized_code) < best_len:
        best_len = len(optimized_code)
        best_code = optimized_code

    # Return as list of lists
    return [list(t) for t in best_code]
