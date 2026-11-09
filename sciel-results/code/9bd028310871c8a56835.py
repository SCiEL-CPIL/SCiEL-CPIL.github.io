import time
from math import gcd

def construct():
    """
    Finds a set of distinct pairwise coprime integers in 1..511 that maximizes
    the sum of 1/(512 - a) using an exact Branch-and-Bound algorithm for
    Maximum Weight Clique with greedy coloring bounds.
    """
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    N = 512
    
    # 1. Precomputation
    # Get all primes < 512
    def get_primes(n):
        sieve = [True] * n
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(N)
    
    # Factor masks for coprimality check
    def get_factor_mask(n):
        mask = 0
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                mask |= (1 << p) # Use prime value as bit index? No, primes are up to 509. 1<<509 is huge but Python handles big ints.
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            mask |= (1 << temp)
        return mask

    # Better: Just store the set of prime factors or a bitmask of indices
    # Since we need to check gcd(a,b)==1, we can just check if they share a prime factor.
    # Let's use a bitmask of prime indices.
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    def get_prime_factor_mask(n):
        mask = 0
        temp = n
        for i, p in enumerate(primes):
            if p * p > temp:
                break
            if temp % p == 0:
                mask |= (1 << i)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            mask |= (1 << prime_to_idx[temp])
        return mask

    factor_masks = [0] * N
    for a in range(1, N):
        factor_masks[a] = get_prime_factor_mask(a)
        
    # Weights
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)
        
    # Build Compatibility Graph (Edges if gcd==1)
    # Nodes are 1..511.
    # We will reorder nodes by weight descending for better B&B performance.
    nodes = list(range(1, N))
    nodes.sort(key=lambda x: -weights[x])
    
    # Map old node ID to new index
    old_to_new = {old: new for new, old in enumerate(nodes)}
    new_to_old = nodes
    
    n_nodes = len(nodes)
    
    # Adjacency masks in the new order
    adj_mask = [0] * n_nodes
    for i in range(n_nodes):
        a = new_to_old[i]
        mask = 0
        for j in range(n_nodes):
            if i == j: continue
            b = new_to_old[j]
            if (factor_masks[a] & factor_masks[b]) == 0:
                mask |= (1 << j)
        adj_mask[i] = mask
        
    # Weights in new order
    w_new = [weights[new_to_old[i]] for i in range(n_nodes)]
    
    # Helper to get top k weights sum from a mask
    def get_top_k_sum(mask, k):
        if mask == 0 or k <= 0:
            return 0.0
        # Extract weights
        ws = []
        m = mask
        i = 0
        while m:
            if m & 1:
                ws.append(w_new[i])
            m >>= 1
            i += 1
        if len(ws) <= k:
            return sum(ws)
        # Sort descending and take top k
        ws.sort(reverse=True)
        return sum(ws[:k])
        
    # Helper to get greedy coloring bound (number of colors)
    def get_greedy_color_count(mask):
        if mask == 0:
            return 0
        count = 0
        m = mask
        while m:
            count += 1
            # Find a maximal independent set in the subgraph induced by m
            # In the compatibility graph, an independent set is a set of pairwise non-coprime numbers?
            # No, we are coloring the compatibility graph.
            # A color class is an independent set in the compatibility graph.
            # Independent set in G_comp means a set of nodes with NO edges between them in G_comp.
            # Edge in G_comp means gcd==1.
            # So no edges means gcd>1 for all pairs?
            # Yes.
            # So we need to find a set of nodes in m that are pairwise NOT coprime.
            # This is a Max Clique in the Conflict Graph.
            # Greedy: Pick a node v. Remove v and all nodes coprime to v (neighbors in G_comp).
            # The remaining nodes are not coprime to v? No, they are non-neighbors.
            # Non-neighbors in G_comp are nodes with gcd>1.
            # So if we pick v, we can keep all u such that gcd(u,v)>1.
            # And we must ensure the set is pairwise non-coprime?
            # Greedy independent set in G_comp:
            # Pick v. Add to set.
            # Remove v and its neighbors (coprime nodes) from m.
            # The remaining nodes are non-coprime to v.
            # But are they pairwise non-coprime? No.
            # This greedy method finds a maximal independent set?
            # Yes, if we simply remove neighbors, the remaining set is an independent set?
            # No. If u and w are both non-neighbors of v, they might be neighbors of each other.
            # So this simple removal does NOT guarantee the set is an independent set.
            
            # Correct Greedy Coloring:
            # We want to partition m into independent sets.
            # Greedy:
            # 1. Pick a node v from m.
            # 2. This node v takes a new color? No, we are constructing the colors.
            # 3. Let's just use a simple upper bound on clique size:
            #    The size of a maximum clique in G is <= the size of a minimum vertex cover? No.
            #    It is <= the chromatic number.
            #    A fast approximation for chromatic number:
            #    Iterate:
            #      Find a maximal independent set I in G[m].
            #      count += 1
            #      m = m \ I
            #    To find Maximal Independent Set in G_comp (where edges are coprime):
            #    Pick v.
            #    I = {v}.
            #    Candidates = m \ {v} \ N(v).
            #    While Candidates:
            #       Pick u from Candidates.
            #       I.add(u)
            #       Candidates = Candidates \ {u} \ N(u)
            #    This I is an independent set.
            
            m_temp = m
            I = 0
            # Pick first node
            # Find lowest set bit
            v = (m_temp & -m_temp).bit_length() - 1
            I |= (1 << v)
            m_temp &= ~(1 << v)
            # Remove neighbors of v
            m_temp &= ~adj_mask[v]
            
            # Continue picking from m_temp
            while m_temp:
                # Pick next node
                u = (m_temp & -m_temp).bit_length() - 1
                I |= (1 << u)
                m_temp &= ~(1 << u)
                m_temp &= ~adj_mask[u]
            
            # I is an independent set.
            m &= ~I
            
        return count

    best_w = 0.0
    best_set = []
    
    def expand(R_w, R_mask, P_mask):
        nonlocal best_w, best_set
        
        if time.time() > deadline:
            return
            
        if P_mask == 0:
            if R_w > best_w:
                best_w = R_w
                # Reconstruct set from R_mask
                s = []
                m = R_mask
                i = 0
                while m:
                    if m & 1:
                        s.append(new_to_old[i])
                    m >>= 1
                    i += 1
                best_set = s
            return
        
        # Bound
        k = get_greedy_color_count(P_mask)
        top_sum = get_top_k_sum(P_mask, k)
        if R_w + top_sum <= best_w + 1e-12:
            return
            
        # Get nodes in P sorted by weight desc
        # Since w_new is sorted desc, we just iterate i from 0 to n_nodes-1
        # and check if bit i is in P_mask
        # But we want to branch on the highest weight node first.
        
        # Find the node v in P_mask with highest weight (lowest index)
        v = -1
        m = P_mask
        i = 0
        while m:
            if m & 1:
                v = i
                break
            m >>= 1
            i += 1
            
        if v == -1:
            return
            
        # Branch 1: Include v
        # New P = P_mask \ {v} \ N(v)
        P_include = P_mask & adj_mask[v] & ~(1 << v)
        # Note: adj_mask[v] contains neighbors (coprime).
        # We want to keep nodes that are coprime to v.
        # So P_include should be P_mask & adj_mask[v] (excluding v).
        # Wait, if we include v, we can only add nodes coprime to v.
        # So P_include = (P_mask & adj_mask[v]) & ~(1 << v)
        
        expand(R_w + w_new[v], R_mask | (1 << v), P_include)
        
        # Branch 2: Exclude v
        P_exclude = P_mask & ~(1 << v)
        expand(R_w, R_mask, P_exclude)
        
    # Initial P is all nodes
    initial_P = (1 << n_nodes) - 1
    expand(0.0, 0, initial_P)
    
    return best_set

# Fallback if best_set is empty (should not happen)
if not best_set:
    # Return primes as fallback
    def _get_primes(n):
        sieve = [True] * n
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    best_set = _get_primes(512)

return best_set
