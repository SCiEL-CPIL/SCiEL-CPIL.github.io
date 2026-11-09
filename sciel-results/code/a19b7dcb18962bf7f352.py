import time
import math
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    N = 512
    # Candidates: 1..511. 
    # Weight for candidate a is 1/(512 - a).
    
    # 1. Precompute prime factors and weights
    primes = []
    for i in range(2, 512):
        is_p = True
        for p in primes:
            if p * p > i:
                break
            if i % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(i)
    
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Bitmask of prime factors for each number
    factor_masks = [0] * 512
    for a in range(1, 512):
        x = a
        mask = 0
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                mask |= (1 << prime_to_idx[p])
                while x % p == 0:
                    x //= p
        if x > 1:
            mask |= (1 << prime_to_idx[x])
        factor_masks[a] = mask
    
    candidates = list(range(1, 512))
    weights = {a: 1.0 / (N - a) for a in candidates}
    
    # Build Conflict Graph
    # Nodes are candidates 1..511.
    # Edge between u, v if gcd(u, v) > 1.
    # We want Maximum Weight Independent Set.
    
    # Adjacency list
    adj = {a: set() for a in candidates}
    
    for i in range(len(candidates)):
        u = candidates[i]
        for j in range(i + 1, len(candidates)):
            v = candidates[j]
            if factor_masks[u] & factor_masks[v]:
                adj[u].add(v)
                adj[v].add(u)
    
    # Precompute degrees
    degrees = {a: len(adj[a]) for a in candidates}
    
    best_score = 0.0
    best_set = set()
    
    # Branch and Reduce for MWIS
    # State: 
    #   candidates_remaining: set of nodes still available
    #   current_score: float
    #   current_set: set of chosen nodes
    
    # To make it efficient, we can use a recursive function that picks a vertex with high degree 
    # or high weight to branch on.
    
    # A simple upper bound for the remaining graph:
    # Sum of weights of all remaining nodes (weak).
    # Better bound: Coloring bound. If we can color the conflict graph with k colors, 
    # the independent set can take at most the sum of the max weight in each color class? No.
    # Actually, for MWIS, a common bound is:
    # For a vertex v, if we don't pick v, we can pick from N(v).
    # If we pick v, we remove N(v) and v.
    
    # Let's use a simpler but effective heuristic search with strong pruning:
    # 1. Greedy initial solution to set a good lower bound.
    
    def greedy_init():
        # Sort by weight descending, then by degree ascending (prefer less connected)
        # Actually, standard greedy: pick highest weight, remove neighbors, repeat.
        avail = set(candidates)
        score = 0.0
        selected = set()
        while avail:
            # Pick node with max weight / (degree+1) or just max weight
            # Let's try max weight
            best_node = max(avail, key=lambda x: weights[x])
            score += weights[best_node]
            selected.add(best_node)
            avail.remove(best_node)
            # Remove neighbors
            for nb in list(adj[best_node]):
                if nb in avail:
                    avail.remove(nb)
        return score, selected
    
    init_score, init_set = greedy_init()
    best_score = init_score
    best_set = init_set
    
    # Another greedy: sort by weight/degree
    def greedy_init_v2():
        avail = set(candidates)
        score = 0.0
        selected = set()
        while avail:
            best_node = max(avail, key=lambda x: weights[x] / (degrees[x] + 1))
            score += weights[best_node]
            selected.add(best_node)
            avail.remove(best_node)
            for nb in list(adj[best_node]):
                if nb in avail:
                    avail.remove(nb)
        return score, selected
        
    s2, sel2 = greedy_init_v2()
    if s2 > best_score:
        best_score = s2
        best_set = sel2

    # Branch and Bound
    # We will use a recursive DFS.
    # To handle the state efficiently, we pass a set of available nodes.
    # Since 511 nodes is small, sets are fine.
    
    # Optimization: If the sum of weights of all available nodes + current_score <= best_score, prune.
    # Precompute total weight? No, it changes.
    # We can maintain a running sum or just calculate it. Calculating sum(avail) is O(N).
    
    def dfs(avail, current_score, current_set):
        nonlocal best_score, best_set
        
        if time.time() > deadline:
            return
            
        if not avail:
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
            return
            
        # Upper bound: current_score + sum(weights[u] for u in avail)
        # This is a weak bound but cheap to compute if we keep a sum.
        # Let's pass the sum as an argument to avoid recalculating.
        # Actually, let's just calculate it for now, N=511 is small.
        bound = current_score + sum(weights[u] for u in avail)
        if bound <= best_score:
            return
            
        # Heuristic: Pick a node to branch on.
        # Strategy: Pick node with highest degree in the induced subgraph of avail?
        # Or highest weight?
        # High degree reduces the tree depth faster (removes more neighbors).
        
        # Calculate degrees within avail
        # This is O(N^2) per node. 511^2 ~ 260k. Might be slow if called many times.
        # Alternative: Pick node with highest weight.
        # Let's try picking node with highest weight first, as it contributes most to the score.
        # If weights are similar, degree matters more.
        
        # Let's pick the node with the highest weight.
        v = max(avail, key=lambda x: weights[x])
        
        # Option 1: Include v
        # New avail: avail - {v} - neighbors(v)
        new_avail_1 = set(avail)
        new_avail_1.remove(v)
        for nb in adj[v]:
            if nb in new_avail_1:
                new_avail_1.remove(nb)
        
        dfs(new_avail_1, current_score + weights[v], current_set | {v})
        
        # Option 2: Exclude v
        # New avail: avail - {v}
        new_avail_2 = set(avail)
        new_avail_2.remove(v)
        
        dfs(new_avail_2, current_score, current_set)
        
    # Start DFS
    # To speed up, we can sort the initial avail? No, set is unordered.
    # The max() call finds the best candidate each time.
    
    # One potential issue: recursion depth. Max depth 511. Python default is 1000. OK.
    
    # Let's run the DFS
    # Initial avail is all candidates
    initial_avail = set(candidates)
    
    # To make the first few steps faster, we can force the inclusion of the top few heavy nodes?
    # The greedy already did a good job. The DFS explores alternatives.
    
    # We need to be careful with the bound. 
    # sum(weights[u] for u in avail) is O(N).
    
    # Let's try to improve the bound using a simple greedy packing bound?
    # Or just rely on the fact that the graph is sparse?
    # Conflict graph density: 
    # Most numbers share small factors. 
    # 1 is coprime to all.
    # Primes are coprime to all non-multiples.
    # The graph is likely connected and dense enough that the bound prunes well?
    # Actually, if the graph is dense, the independent set is small, so the tree is shallow?
    # If the graph is sparse, the independent set is large, tree is deep.
    
    # Let's run it.
    try:
        dfs(initial_avail, 0.0, set())
    except RecursionError:
        # Fallback if recursion depth exceeded (unlikely with 511)
        pass

    # Convert best_set to list
    return list(best_set)

# Test locally
if __name__ == "__main__":
    res = construct()
    print(f"Selected: {res}")
    score = sum(1.0 / (512 - a) for a in res)
    print(f"Score: {score}")
    # Verify pairwise coprime
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            if math.gcd(res[i], res[j]) != 1:
                print(f"ERROR: {res[i]} and {res[j]} not coprime")
                valid = False
    if valid:
        print("Valid")
    print("Done")
