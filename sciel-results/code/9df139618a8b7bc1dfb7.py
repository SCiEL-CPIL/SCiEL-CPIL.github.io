import time
import random
import math

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Generate standard superpermutation for n-1 to get the permutation list
    def get_standard_superperm(m):
        word = [1]
        for k in range(2, m + 1):
            perms, seen = [], set()
            for i in range(len(word) - k + 2):
                window = tuple(word[i:i + k - 1])
                if len(set(window)) == k - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [k] + list(p)
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word
    
    # Get permutations of n-1 symbols in standard order
    word_prev = get_standard_superperm(n - 1)
    
    perms_order = []
    seen = set()
    for i in range(len(word_prev) - (n - 1) + 1):
        window = tuple(word_prev[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms_order.append(window)
    
    N = len(perms_order)  # (n-1)! = 720 for n=7
    
    # Precompute overlap matrix
    # overlap[i][j] = max k such that perms_order[i][n-1-k:] == perms_order[j][:k]
    max_overlap_len = n - 2  # Maximum possible overlap is n-2
    
    # Build overlap matrix efficiently
    overlap = [[0] * N for _ in range(N)]
    for i in range(N):
        pi = perms_order[i]
        for j in range(N):
            if i == j:
                overlap[i][j] = 0
                continue
            pj = perms_order[j]
            # Check from max possible down to 0
            for k in range(max_overlap_len, 0, -1):
                if pi[n-1-k:] == pj[:k]:
                    overlap[i][j] = k
                    break
    
    # Current order: identity
    order = list(range(N))
    
    def calc_total_overlap(perm_order):
        total = 0
        for i in range(N - 1):
            total += overlap[perm_order[i]][perm_order[i+1]]
        return total
    
    current_overlap = calc_total_overlap(order)
    best_overlap = current_overlap
    best_order = order[:]
    
    # For efficient 2-opt: precompute nothing extra, but compute delta quickly
    # When reversing segment [i, j] (0-indexed positions in order):
    # Old edges affected: (i-1, i), (i, i+1), ..., (j-1, j), (j, j+1)
    # New edges: (i-1, j), (j, j-1), ..., (i+1, i), (i, j+1)
    # Delta = [overlap[order[i-1]][order[j]] + overlap[order[i]][order[j+1]] if in bounds
    #         + sum_{k=i}^{j-1} overlap[order[k+1]][order[k]]]
    #         - [overlap[order[i-1]][order[i]] + overlap[order[j]][order[j+1]] if in bounds
    #            + sum_{k=i}^{j-1} overlap[order[k]][order[k+1]]]
    
    def get_delta_reverse(i, j):
        """Compute delta overlap when reversing positions i..j in order."""
        if i >= j:
            return 0
        
        delta = 0
        
        # Edge before segment: (i-1) -> i becomes (i-1) -> j
        if i > 0:
            delta += overlap[order[i-1]][order[j]] - overlap[order[i-1]][order[i]]
        
        # Edge after segment: j -> (j+1) becomes i -> (j+1)
        if j < N - 1:
            delta += overlap[order[i]][order[j+1]] - overlap[order[j]][order[j+1]]
        
        # Internal edges: k -> k+1 becomes k+1 -> k for k in [i, j-1]
        for k in range(i, j):
            delta += overlap[order[k+1]][order[k]] - overlap[order[k]][order[k+1]]
        
        return delta
    
    # Simulated annealing with 2-opt moves
    # Also try swaps as a simpler move
    
    def get_delta_swap(p, q):
        """Compute delta overlap when swapping positions p and q."""
        if p == q:
            return 0
        if q < p:
            p, q = q, p
        
        delta = 0
        u = order[p]
        v = order[q]
        
        # Edges affected: (p-1,p), (p,p+1), (q-1,q), (q,q+1)
        # But if p and q are adjacent, it's simpler
        
        # Let's just compute old and new contributions
        # Old:
        old_cost = 0
        if p > 0:
            old_cost += overlap[order[p-1]][u]
        if p < N - 1:
            old_cost += overlap[u][order[p+1]]
        if q > 0:
            old_cost += overlap[order[q-1]][v]
        if q < N - 1:
            old_cost += overlap[v][order[q+1]]
        
        # New (after swap):
        new_cost = 0
        if p > 0:
            new_cost += overlap[order[p-1]][v]
        if p < N - 1:
            new_cost += overlap[v][order[p+1]]
        if q > 0:
            new_cost += overlap[order[q-1]][u]
        if q < N - 1:
            new_cost += overlap[u][order[q+1]]
        
        # Handle the edge between p and q if they're adjacent
        if q == p + 1:
            # Old: ... u -> v ...
            # New: ... v -> u ...
            # The edge p->p+1 is u->v, becomes v->u
            # Already accounted above? Let me re-examine.
            # Actually when p and q are adjacent, the edges (p,p+1) and (q-1,q) are the same edge.
            # So we double-counted. Let me redo this more carefully.
            pass
        
        return new_cost - old_cost
    
    # Simpler approach: just try random 2-opt reversals and swaps, accept if improves
    # Use simulated annealing to escape local optima
    
    temp = 10.0
    cooling = 0.999
    
    # First, do a quick greedy pass to get a good starting point
    # Then simulated annealing
    
    # Actually, let's just do many random 2-opt moves with acceptance
    iterations = 0
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        # Choose a random move
        if random.random() < 0.5:
            # 2-opt: reverse a segment
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            delta = get_delta_reverse(i, j)
            
            # Apply
            order[i:j+1] = reversed(order[i:j+1])
            current_overlap += delta
            iterations += 1
            
            if delta > 0:
                if current_overlap > best_overlap:
                    best_overlap = current_overlap
                    best_order = order[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Accept with probability exp(delta/temp)
                if random.random() < math.exp(delta / temp):
                    no_improve_count += 1
                else:
                    # Reject: undo
                    order[i:j+1] = reversed(order[i:j+1])
                    current_overlap -= delta
                    no_improve_count += 1
        else:
            # Swap two positions
            p = random.randint(0, N - 1)
            q = random.randint(0, N - 1)
            if p == q:
                continue
            if q < p:
                p, q = q, p
            
            u = order[p]
            v = order[q]
            
            # Compute delta for swap
            # Edges that change:
            # If p and q are not adjacent:
            #   (p-1,p), (p,p+1), (q-1,q), (q,q+1)
            # If adjacent (q = p+1):
            #   (p-1,p), (p,p+1)=same as (q-1,q), (q,q+1)
            
            delta = 0
            if p > 0:
                delta += overlap[order[p-1]][v] - overlap[order[p-1]][u]
            if p < N - 1:
                delta += overlap[v][order[p+1]] - overlap[u][order[p+1]]
            if q > 0:
                delta += overlap[order[q-1]][u] - overlap[order[q-1]][v]
            if q < N - 1:
                delta += overlap[u][order[q+1]] - overlap[v][order[q+1]]
            
            # If adjacent, we double-counted the middle edge
            if q == p + 1:
                # The edge p->p+1 was u->v, now it's v->u
                # In the above, for p: we added overlap[v][order[p+1]] - overlap[u][order[p+1]] = overlap[v][v] - overlap[u][v] = 0 - overlap[u][v]
                # For q: we added overlap[order[q-1]][u] - overlap[order[q-1]][v] = overlap[u][u] - overlap[u][v] = 0 - overlap[u][v]
                # So we subtracted overlap[u][v] twice, but it should only be subtracted once.
                # And we need to add overlap[v][u] once.
                # Net correction: + overlap[u][v] + overlap[v][u] - ... hmm this is getting messy.
                # Let me just recompute from scratch for adjacent case.
                delta = 0
                if p > 0:
                    delta += overlap[order[p-1]][v] - overlap[order[p-1]][u]
                if q < N - 1:
                    delta += overlap[u][order[q+1]] - overlap[v][order[q+1]]
                # Middle edge: was u->v, now v->u
                delta += overlap[v][u] - overlap[u][v]
            
            # Apply swap
            order[p], order[q] = v, u
            current_overlap += delta
            iterations += 1
            
            if delta > 0:
                if current_overlap > best_overlap:
                    best_overlap = current_overlap
                    best_order = order[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                if random.random() < math.exp(delta / temp):
                    no_improve_count += 1
                else:
                    order[p], order[q] = u, v
                    current_overlap -= delta
                    no_improve_count += 1
        
        temp *= cooling
        
        if no_improve_count > max_no_improve:
            # Restart with best solution
            order = best_order[:]
            current_overlap = best_overlap
            no_improve_count = 0
            temp = 10.0
    
    # Use the best order found
    order = best_order
    
    # Construct the final word
    word = []
    for idx in range(N):
        p = perms_order[order[idx]]
        block = list(p) + [n] + list(p)
        if not word:
            word = block
        else:
            found = 0
            max_k = n - 2
            for k in range(max_k, 0, -1):
                if word[-k:] == block[:k]:
                    found = k
                    break
            word += block[found:]
    
    return "".join(map(str, word))
