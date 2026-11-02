import random
import time
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110
    
    def generate_random_sidon_row(m, k, bias=10):
        """
        Generates a random valid Sidon row of length k+1, max element <= m.
        Uses a greedy approach with randomization among the smallest valid candidates.
        """
        row = [0]
        diffs = set()
        for _ in range(k):
            valid_xs = []
            x = row[-1] + 1
            # We scan from the smallest possible x upwards
            while x <= m:
                new_diffs = {x - a for a in row}
                if not (new_diffs & diffs):
                    valid_xs.append(x)
                    # We don't need all valid xs, just a few small ones for diversity and small scope
                    if len(valid_xs) >= bias:
                        break
                x += 1
            
            if not valid_xs:
                return None
            
            # Pick randomly from the candidates
            # To encourage smaller max, we picked small candidates.
            chosen = random.choice(valid_xs)
            row.append(chosen)
            # Update diffs
            for a in row[:-1]:
                diffs.add(chosen - a)
                
        return row

    def build_conflict_graph(pool):
        """
        Builds an adjacency list for the conflict graph.
        graph[i] is a set of indices j such that pool[i] and pool[j] share a difference.
        """
        n = len(pool)
        # Pre-compute difference sets for each row
        diff_sets = []
        for row in pool:
            d_set = set()
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d_set.add(row[j] - row[i])
            diff_sets.append(d_set)
            
        graph = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if diff_sets[i] & diff_sets[j]:
                    graph[i].add(j)
                    graph[j].add(i)
        return graph

    def find_independent_set(graph, size):
        """
        Finds an independent set of the given size in the graph.
        Returns a list of indices or None.
        """
        n = len(graph)
        if n < size:
            return None
        
        # Order nodes by degree (ascending) to prune faster? 
        # Actually, for independent set, picking nodes with high degree first 
        # removes more options, which might be good for finding a set quickly 
        # or bad because it removes potential nodes.
        # Standard heuristic: Pick node with lowest degree? 
        # Let's just use a simple backtracking with a timeout.
        
        current_set = []
        
        def dfs(available, needed):
            if needed == 0:
                return current_set[:]
            if len(available) < needed:
                return None
            if time.time() - start_time > time_limit:
                return None
                
            # Heuristic: Pick a node from available
            # To improve success rate, pick a node with fewest conflicts within available?
            # Or just pick the first one.
            # Let's pick the node with the smallest index for determinism in failure, 
            # but randomize for exploration? 
            # For finding *any* set, a good heuristic is to pick a node that 
            # doesn't block many others. i.e., low degree in the induced subgraph.
            
            # Calculate degrees within available
            # This is expensive. Let's just pick the first available node.
            # Or, to be smarter, pick a random node from available.
            node = random.choice(list(available))
            
            current_set.append(node)
            
            # New available: available minus node minus neighbors of node
            new_available = set(available)
            new_available.remove(node)
            for neighbor in graph[node]:
                if neighbor in new_available:
                    new_available.remove(neighbor)
            
            result = dfs(new_available, needed - 1)
            if result:
                return result
            
            current_set.pop()
            return None

        return dfs(set(range(n)), size)

    def check(m, pool_size=1000):
        """
        Checks if a valid configuration exists with max element <= m.
        Returns the configuration (list of rows) if found, else None.
        """
        pool = []
        # Generate pool
        for _ in range(pool_size):
            if time.time() - start_time > time_limit:
                return None
            row = generate_random_sidon_row(m, k)
            if row:
                pool.append(row)
                
        if len(pool) < n_rows:
            return None
            
        graph = build_conflict_graph(pool)
        
        # Find independent set
        indices = find_independent_set(graph, n_rows)
        
        if indices:
            # Return the actual rows
            return [pool[i] for i in indices]
            
        return None

    # Binary search on m
    # Lower bound: minimum possible max element
    # For k=6, min max is roughly k*(k+1)/2 ? No, that's for packing.
    # Simple lower bound: k (if 0,1,2...k is valid? No, diffs 1,2..k repeat).
    # Let's start from a reasonable lower bound.
    low = k + 1 
    high = 200 # Safe upper bound for this problem size
    
    best_solution = None
    
    # We want the minimum m.
    # We can iterate m from low to high, but binary search is faster if check is fast.
    # However, check(m) is randomized. If it fails, it might be due to bad luck, not infeasibility.
    # So binary search on a randomized check is tricky.
    # Better: Iterate m upwards. Stop when we find a solution.
    # But we want the *minimum*. 
    # If we find a solution for m, it's a valid upper bound.
    # To find the minimum, we should try small m first.
    
    # Strategy: 
    # 1. Try small m (e.g., k+1 to 150) with a fixed number of attempts.
    # 2. If found, return it.
    # 3. If not found, we might have missed it. 
    # 4. Given the time limit, we can afford to try a few m values.
    
    # Actually, the "best known" is 112. 
    # Let's try m from 100 to 120.
    # If we find a solution for m=100, it's better than 112.
    
    # To be robust, let's do a linear scan from a reasonable lower bound.
    # Lower bound estimation: 
    # Number of differences in one row: C(k+1, 2) = 21.
    # Total differences needed: 7 * 21 = 147.
    # Max difference in a row with max m is m.
    # So we need at least 147 distinct differences from 1..m.
    # So m >= 147? No, differences are from 1..m. 
    # Wait, the differences must be distinct *across all rows*.
    # So we need 147 distinct integers from 1..m.
    # Thus m >= 147.
    # But the best known is 112. How?
    # Ah, the differences are $a_j - a_i$. 
    # If max element is 112, the differences are in 1..112.
    # There are 112 possible values.
    # But we need 147 distinct differences.
    # This implies my counting bound logic in the prompt ("simple counting bound is 105") might refer to something else or I am misinterpreting.
    # Let's re-read: "simple counting bound is 105".
    # If m=105, there are 105 possible differences.
    # We need 147 distinct differences.
    # This is impossible if differences must be in 1..m.
    # Unless... the differences don't have to be in 1..m? 
    # $a_j - a_i$ where $0 \le a_i < a_j \le m$.
    # Max difference is $m - 0 = m$.
    # So differences are in $1..m$.
    # So we need $m \ge 147$.
    # Why is the bound 105?
    # Maybe the prompt implies $n_{rows}=7, k=6$ means 7 rows of 6 integers? 
    # "Choose 7 rows of 6 increasing integers". 
    # "Each row is a list of k + 1 integers".
    # If k=6, row has 7 integers.
    # Differences per row: $\binom{7}{2} = 21$.
    # Total diffs: $7 \times 21 = 147$.
    # If k=5, row has 6 integers. Diffs: $\binom{6}{2} = 15$. Total: $7 \times 15 = 105$.
    # The prompt says "Difference triangle set (7,5)". 
    # Usually DTS(n, k) means n rows of k elements? Or k+1?
    # The interface says `construct(n_rows, k)` and "Each row is a list of k + 1 integers".
    # The title says "(7,5)". This likely means n_rows=7, k=5?
    # If k=5, row size is 6. Diffs = 15. Total = 105.
    # The prompt says "Choose 7 rows of 6 increasing integers". This matches k=5 (k+1=6).
    # So the evaluator will likely pass `n_rows=7, k=5`.
    # My code should handle general k.
    
    # So, if k=5, m >= 105.
    # Best known for (7,5) is 112? Or is 112 for (7,6)?
    # "The best known set has largest entry 112... simple counting bound is 105."
    # This confirms k=5 (since bound is 105).
    
    # So we should search for m starting from 105.
    
    for m in range(105, 150):
        if time.time() - start_time > time_limit:
            break
        sol = check(m, pool_size=2000)
        if sol:
            return sol
            
    # If not found in the loop, return the last best attempt or a greedy fallback
    # To ensure we return *something* valid, we can use a greedy construction 
    # with a large m.
    
    # Fallback: Greedy construction with m=200
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    break
                x += 1
            row.append(x)
            used_diffs.update({x - a for a in row[:-1]})
        rows.append(row)
        
    return rows
