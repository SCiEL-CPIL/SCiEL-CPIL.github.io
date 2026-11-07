"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from itertools import combinations
from collections import defaultdict

def construct():
    """
    Construct 14 distinct positive integers <= 10000 and exactly 35 edges
    to minimize max(number of distinct edge sums, number of distinct edge products).
    
    Strategy: Constraint satisfaction search using Union-Find to maintain
    connected components of edges sharing identical sums or products.
    Directly backtracks to construct 35-edge subgraphs within target score K.
    """
    
    n = 14
    m = 35
    max_val = 10000
    all_pairs = list(combinations(range(n), 2))
    
    class UnionFind:
        def __init__(self, size):
            self.parent = list(range(size))
            self.rank = [0] * size
        
        def find(self, x):
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x
        
        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)
            if rx == ry:
                return False
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            return True
    
    def score(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            s = values[i] + values[j]
            p = values[i] * values[j]
            sums.add(s)
            prods.add(p)
        return max(len(sums), len(prods))
    
    def try_construct(values, target_k, max_attempts=200):
        """
        Try to construct exactly m edges such that distinct sums <= K and distinct products <= K.
        Uses greedy selection with sum/product grouping.
        """
        # Precompute all possible sums and products for each edge
        edge_sums = {}
        edge_prods = {}
        for i, j in all_pairs:
            s = values[i] + values[j]
            p = values[i] * values[j]
            edge_sums[(i, j)] = s
            edge_prods[(i, j)] = p
        
        # Group edges by sum and by product
        sum_to_edges = defaultdict(list)
        prod_to_edges = defaultdict(list)
        for i, j in all_pairs:
            sum_to_edges[edge_sums[(i, j)]].append((i, j))
            prod_to_edges[edge_prods[(i, j)]].append((i, j))
        
        # Get unique sums and products sorted by frequency
        unique_sums = sorted(sum_to_edges.keys(), key=lambda s: -len(sum_to_edges[s]))
        unique_prods = sorted(prod_to_edges.keys(), key=lambda p: -len(prod_to_edges[p]))
        
        best_edges = None
        
        for attempt in range(max_attempts):
            # Select K sums and K products
            if attempt == 0:
                selected_sums = set(unique_sums[:target_k])
                selected_prods = set(unique_prods[:target_k])
            else:
                # Randomized selection
                k_s = max(1, target_k - random.randint(0, 3))
                k_p = max(1, target_k - random.randint(0, 3))
                selected_sums = set(unique_sums[:k_s])
                selected_prods = set(unique_prods[:k_p])
                
                if len(selected_sums) < target_k:
                    remaining_sums = [s for s in unique_sums if s not in selected_sums]
                    if remaining_sums:
                        selected_sums.update(random.sample(remaining_sums, min(len(remaining_sums), target_k - len(selected_sums))))
                if len(selected_prods) < target_k:
                    remaining_prods = [p for p in unique_prods if p not in selected_prods]
                    if remaining_prods:
                        selected_prods.update(random.sample(remaining_prods, min(len(remaining_prods), target_k - len(selected_prods))))
            
            # Find all edges whose sum is in selected_sums AND product is in selected_prods
            valid_edges = []
            for i, j in all_pairs:
                s = edge_sums[(i, j)]
                p = edge_prods[(i, j)]
                if s in selected_sums and p in selected_prods:
                    valid_edges.append((i, j))
            
            if len(valid_edges) < m:
                continue
            
            # Greedy selection: prefer edges that reuse existing sums/products
            selected_edges = []
            used_sums = set()
            used_prods = set()
            
            # Shuffle for diversity, then sort by priority
            random.shuffle(valid_edges)
            
            def edge_priority(edge):
                i, j = edge
                s = edge_sums[edge]
                p = edge_prods[edge]
                priority = 0
                if s in used_sums:
                    priority += 10
                if p in used_prods:
                    priority += 10
                return -priority
            
            # Iteratively add edges
            remaining = valid_edges[:]
            while len(selected_edges) < m and remaining:
                remaining.sort(key=edge_priority)
                edge = remaining.pop(0)
                selected_edges.append(edge)
                used_sums.add(edge_sums[edge])
                used_prods.add(edge_prods[edge])
            
            if len(selected_edges) == m:
                # Verify the score
                actual_sums = set()
                actual_prods = set()
                for i, j in selected_edges:
                    actual_sums.add(edge_sums[(i, j)])
                    actual_prods.add(edge_prods[(i, j)])
                
                actual_score = max(len(actual_sums), len(actual_prods))
                if actual_score <= target_k:
                    best_edges = [list(e) for e in selected_edges]
                    break
        
        return best_edges
    
    def local_search(values, edges, max_iter=3000):
        """Improve solution via local search."""
        current_score = score(values, edges)
        best_score = current_score
        best_values = values[:]
        best_edges = edges[:]
        
        for iteration in range(max_iter):
            move_type = random.choice(['swap_val', 'add_remove_edge', 'swap_edge'])
            
            new_values = values[:]
            new_edges = edges[:]
            
            if move_type == 'swap_val':
                i, j = random.sample(range(n), 2)
                new_values[i], new_values[j] = new_values[j], new_values[i]
                
            elif move_type == 'add_remove_edge':
                existing_set = set(tuple(sorted(e)) for e in new_edges)
                available = [e for e in all_pairs if tuple(sorted(e)) not in existing_set]
                if not available:
                    continue
                new_e = random.choice(available)
                rem_e = random.choice(new_edges)
                new_edges.remove(rem_e)
                new_edges.append(list(new_e))
                
            elif move_type == 'swap_edge':
                if len(new_edges) < 2:
                    continue
                existing_set = set(tuple(sorted(e)) for e in new_edges)
                available = [e for e in all_pairs if tuple(sorted(e)) not in existing_set]
                if len(available) < 2:
                    continue
                e1, e2 = random.sample(new_edges, 2)
                new_e1, new_e2 = random.sample(available, 2)
                new_edges.remove(e1)
                new_edges.remove(e2)
                new_edges.append(list(new_e1))
                new_edges.append(list(new_e2))
            
            new_score = score(new_values, new_edges)
            
            if new_score < current_score:
                values = new_values
                edges = new_edges
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_values = values[:]
                    best_edges = edges[:]
            elif new_score == current_score and random.random() < 0.3:
                values = new_values
                edges = new_edges
        
        return best_score, best_values, best_edges
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    # Try increasing target_k from low to high
    for target_k in range(12, 25):
        found = False
        
        # Try multiple value configurations
        for restart in range(15):
            if restart == 0:
                values = list(range(1, n + 1))
            elif restart == 1:
                values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            elif restart == 2:
                primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
                values = primes[:]
            elif restart == 3:
                values = [1, 2, 3, 4, 5, 6, 8, 9, 12, 16, 24, 36, 72, 144]
            elif restart == 4:
                values = sorted(random.sample(range(1, 100), n))
            elif restart == 5:
                values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
            elif restart == 6:
                values = [2 * i for i in range(1, n + 1)]
            elif restart == 7:
                fibs = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
                values = fibs[:]
            elif restart == 8:
                values = sorted(random.sample(range(1, 50), n))
            else:
                if best_values:
                    values = best_values[:]
                    i, j = random.sample(range(n), 2)
                    values[i], values[j] = values[j], values[i]
                else:
                    values = list(range(1, n + 1))
            
            # Ensure distinct and valid
            if len(set(values)) != n or min(values) < 1 or max(values) > max_val:
                continue
            
            # Try to construct edges with target_k
            edges = try_construct(values, target_k)
            
            if edges is not None:
                current_score = score(values, edges)
                if current_score <= target_k:
                    ls_score, ls_values, ls_edges = local_search(values, edges, max_iter=2000)
                    if ls_score < best_score:
                        best_score = ls_score
                        best_values = ls_values
                        best_edges = ls_edges
                    found = True
                    if best_score <= target_k:
                        break
        
        if found and best_score <= target_k:
            break
    
    # Final local search on best solution
    if best_values and best_edges:
        ls_score, ls_values, ls_edges = local_search(best_values, best_edges, max_iter=5000)
        if ls_score < best_score:
            best_score = ls_score
            best_values = ls_values
            best_edges = ls_edges
    
    # Ensure types are correct
    best_values = [int(v) for v in best_values]
    best_edges = [[int(i), int(j)] for i, j in best_edges]
    
    return {'values': best_values, 'edges': best_edges}

# EVOLVE-BLOCK-END
