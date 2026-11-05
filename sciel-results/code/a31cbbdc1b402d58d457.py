import itertools
import random
import math

def construct(n):
    if n < 7:
        return standard_construct(n)
    
    # For n=7, use TSP on permutations of 1..6
    symbols = list(range(1, 7))
    perms = list(itertools.permutations(symbols))
    num_perms = len(perms)
    
    # Precompute overlap matrix
    # overlap[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    overlap = [[0] * num_perms for _ in range(num_perms)]
    for i in range(num_perms):
        pi = perms[i]
        for j in range(num_perms):
            pj = perms[j]
            # Max overlap is 6
            for k in range(6, -1, -1):
                if pi[-k:] == pj[:k]:
                    overlap[i][j] = k
                    break
    
    # Simulated Annealing to maximize total overlap
    # Initial path: random permutation of indices
    current_path = list(range(num_perms))
    random.shuffle(current_path)
    
    def get_total_overlap(path):
        total = 0
        for i in range(len(path) - 1):
            total += overlap[path[i]][path[i+1]]
        return total
    
    current_overlap = get_total_overlap(current_path)
    best_overlap = current_overlap
    best_path = current_path[:]
    
    # SA Parameters
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.01
    max_iterations = 50000 # Adjust based on time limit
    
    while T > min_T and max_iterations > 0:
        # Random move: swap two elements or reverse a segment
        i = random.randint(0, num_perms - 2)
        j = random.randint(i + 1, num_perms - 1)
        
        # Try swapping i and j
        new_path = current_path[:]
        new_path[i], new_path[j] = new_path[j], new_path[i]
        
        # Calculate delta overlap
        # Only edges involving i and j change
        # Edges: (i-1, i), (i, i+1), (j-1, j), (j, j+1)
        # Need to be careful with boundaries and if i, j are adjacent
        
        old_ov = 0
        new_ov = 0
        
        # Helper to get overlap contribution of a pair
        def ov(u, v):
            if u is None or v is None:
                return 0
            return overlap[u][v]
        
        # Calculate old contributions
        if i > 0:
            old_ov += ov(current_path[i-1], current_path[i])
            new_ov += ov(new_path[i-1], new_path[i])
        else:
            # i is 0, no left neighbor
            pass
            
        if i < num_perms - 1:
            old_ov += ov(current_path[i], current_path[i+1])
            new_ov += ov(new_path[i], new_path[i+1])
            
        if j > 0:
            old_ov += ov(current_path[j-1], current_path[j])
            new_ov += ov(new_path[j-1], new_path[j])
            
        if j < num_perms - 1:
            old_ov += ov(current_path[j], current_path[j+1])
            new_ov += ov(new_path[j], new_path[j+1])
            
        # If i and j are adjacent, we double counted or missed some?
        # If j == i+1:
        # Old: (i-1, i), (i, i+1), (i+1, i+2)
        # New: (i-1, i+1), (i+1, i), (i, i+2)
        # My logic above:
        # i > 0: adds (i-1, i) -> new (i-1, i+1)
        # i < n-1: adds (i, i+1) -> new (i, i) -- WAIT.
        # If we swap i and j, and j=i+1:
        # new_path[i] = old_path[j]
        # new_path[j] = old_path[i]
        # Edge (i, i+1) in old becomes (old_i, old_i+1).
        # In new, edge (i, i+1) is (new_i, new_i+1) = (old_i+1, old_i).
        # My logic:
        # i < n-1: old_ov += ov(path[i], path[i+1]). new_ov += ov(new_path[i], new_path[i+1]).
        # j = i+1.
        # j > 0: old_ov += ov(path[j-1], path[j]) = ov(path[i], path[i+1]).
        # new_ov += ov(new_path[j-1], new_path[j]) = ov(new_path[i], new_path[i+1]).
        # So we added (i, i+1) twice to old and new.
        # But the edge (i, i+1) is the one being swapped.
        # Actually, if j=i+1, the set of affected edges is:
        # (i-1, i), (i, i+1), (i+1, i+2)
        # My logic adds:
        # (i-1, i) -> (i-1, new_i)
        # (i, i+1) -> (new_i, new_i+1) [This is the internal edge of the swap]
        # (i, i+1) again from j side? No, j side adds (j-1, j) which is (i, i+1).
        # So yes, double counting.
        
        # Let's simplify: just recalculate total overlap for small n?
        # n=720. 50000 iterations * 720 ops = 36M ops. Might be slow in Python.
        # Better to fix the delta calculation properly.
        
        # Correct Delta Calculation for Swap i, j:
        # Affected edges are those incident to i or j.
        # Let's just compute the change directly.
        
        # Revert to simple calculation if logic is complex, but optimize by only checking neighbors.
        # For n=720, 50k iterations is fine if delta is O(1).
        
        # Let's restart the delta logic cleanly.
        # Old edges:
        # 1. (path[i-1], path[i]) if i>0
        # 2. (path[i], path[i+1]) if i<n-1
        # 3. (path[j-1], path[j]) if j>0
        # 4. (path[j], path[j+1]) if j<n-1
        
        # New edges:
        # 1. (new_path[i-1], new_path[i]) if i>0  -> (path[i-1], path[j])
        # 2. (new_path[i], new_path[i+1]) if i<n-1 -> (path[j], path[i+1])  [if i+1 != j]
        #    If i+1 == j, this edge is (path[j], path[i])
        # 3. (new_path[j-1], new_path[j]) if j>0  -> (path[j-1], path[i])
        # 4. (new_path[j], new_path[j+1]) if j<n-1 -> (path[i], path[j+1])
        
        # Case 1: j > i + 1
        # Old: (i-1, i), (i, i+1), (j-1, j), (j, j+1)
        # New: (i-1, j), (j, i+1), (j-1, i), (i, j+1)
        
        # Case 2: j == i + 1
        # Old: (i-1, i), (i, i+1), (i+1, i+2)
        # New: (i-1, i+1), (i+1, i), (i, i+2)
        
        # Let's implement this specific logic.
        
        delta = 0
        
        if j > i + 1:
            if i > 0:
                delta -= overlap[current_path[i-1]][current_path[i]]
                delta += overlap[current_path[i-1]][current_path[j]]
            if i < num_perms - 1:
                delta -= overlap[current_path[i]][current_path[i+1]]
                delta += overlap[current_path[j]][current_path[i+1]]
            if j > 0:
                delta -= overlap[current_path[j-1]][current_path[j]]
                delta += overlap[current_path[j-1]][current_path[i]]
            if j < num_perms - 1:
                delta -= overlap[current_path[j]][current_path[j+1]]
                delta += overlap[current_path[i]][current_path[j+1]]
        else: # j == i + 1
            if i > 0:
                delta -= overlap[current_path[i-1]][current_path[i]]
                delta += overlap[current_path[i-1]][current_path[i+1]]
            # Middle edge (i, i+1) becomes (i+1, i)
            delta -= overlap[current_path[i]][current_path[i+1]]
            delta += overlap[current_path[i+1]][current_path[i]]
            if j < num_perms - 1: # i+1 < n-1
                delta -= overlap[current_path[j]][current_path[j+1]]
                delta += overlap[current_path[i]][current_path[j+1]]
        
        # Acceptance
        if delta > 0:
            # Always accept improvement
            current_path[i], current_path[j] = current_path[j], current_path[i]
            current_overlap += delta
            if current_overlap > best_overlap:
                best_overlap = current_overlap
                best_path = current_path[:]
        else:
            # Accept with probability
            if random.random() < math.exp(delta / T):
                current_path[i], current_path[j] = current_path[j], current_path[i]
                current_overlap += delta
                
        T *= cooling_rate
        max_iterations -= 1
        
    # Construct string from best_path
    if best_path is None:
        return standard_construct(n)
        
    # Build the word
    word = []
    # First block
    p = perms[best_path[0]]
    block = list(p) + [7] + list(p)
    word.extend(block)
    
    for idx in range(1, len(best_path)):
        p = perms[best_path[idx]]
        block = list(p) + [7] + list(p)
        ov = overlap[best_path[idx-1]][best_path[idx]]
        word.extend(block[ov:])
        
    return "".join(map(str, word))

def standard_construct(n):
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
