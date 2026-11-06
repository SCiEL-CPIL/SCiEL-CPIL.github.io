"""Sidon set extension in [1, 284] containing {1,2,5,11}."""
import time
import random
import sys
from collections import defaultdict, deque

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    seed_elements = [1, 2, 5, 11]
    max_val = 284
    max_sum = 2 * max_val
    
    def compute_sums_bitset(a):
        """Compute bitset of all pairwise sums a[i]+a[j] for i<=j."""
        bits = 0
        n = len(a)
        for i in range(n):
            for j in range(i, n):
                bits |= 1 << (a[i] + a[j])
        return bits

    def compute_sums_dict(a):
        """Compute dict of sum -> list of (i,j) pairs."""
        sums = defaultdict(list)
        n = len(a)
        for i in range(n):
            for j in range(i, n):
                s = a[i] + a[j]
                sums[s].append((i, j))
        return sums

    def can_add(x, a, sums_bits):
        """Check if x can be added to a without creating duplicate sums."""
        if (sums_bits >> (2 * x)) & 1:
            return False
        for y in a:
            s = x + y
            if (sums_bits >> s) & 1:
                return False
        return True

    def greedy_extend_bitset(a, sums_bits):
        """Greedy extension using bitsets. Returns (new_a, new_sums_bits)."""
        a = list(a)
        sums_bits = sums_bits
        for x in range(1, max_val + 1):
            if x in a:
                continue
            if can_add(x, a, sums_bits):
                a.append(x)
                a.sort()
                sums_bits |= 1 << (2 * x)
                for y in a:
                    if y != x:
                        sums_bits |= 1 << (x + y)
        return a, sums_bits

    def max_flow_bipartite(adj, source, sink, num_nodes):
        """
        Max flow on bipartite graph using Ford-Fulkerson with BFS (Edmonds-Karp).
        adj: list of lists, adj[u] = [(v, cap), ...]
        Returns max flow value.
        """
        # Build residual graph
        flow = [[0] * num_nodes for _ in range(num_nodes)]
        
        def bfs():
            # Find augmenting path from source to sink
            parent = [-1] * num_nodes
            visited = [False] * num_nodes
            queue = deque([source])
            visited[source] = True
            
            while queue:
                u = queue.popleft()
                for v, cap in adj[u]:
                    if not visited[v] and cap - flow[u][v] > 0:
                        visited[v] = True
                        parent[v] = u
                        if v == sink:
                            return True
                        queue.append(v)
            return False
        
        total_flow = 0
        while bfs():
            # Find bottleneck
            path_flow = float('inf')
            v = sink
            while v != source:
                u = parent[v]
                path_flow = min(path_flow, adj[u][v][1] - flow[u][v])
                v = u
            
            # Update flows
            v = sink
            while v != source:
                u = parent[v]
                flow[u][v] += path_flow
                flow[v][u] -= path_flow
                v = u
            
            total_flow += path_flow
        
        return total_flow

    def min_cut_resolve(a, sums_dict, seed_set):
        """
        Use min-cut to identify minimum set of non-seed elements to remove
        to resolve all sum conflicts.
        Returns list of elements to remove.
        """
        # Find all conflicting sums
        conflicts = []
        for s, pairs in sums_dict.items():
            if len(pairs) > 1:
                conflicts.append((s, pairs))
        
        if not conflicts:
            return []
        
        # Build bipartite graph:
        # Left side: elements (indices)
        # Right side: sum values
        # Edge: element i is connected to sum s if element i participates in sum s
        # We want minimum vertex cover on the element side only (can't remove sums)
        # Actually, we want to remove minimum elements such that no sum has >1 pair.
        # This is a hitting set / set cover problem, not directly min-cut.
        # 
        # Alternative: For each conflicting sum, at least one of the pairs must be "broken".
        # Breaking a pair (i,j) means removing element i or element j.
        # So we have a 2-SAT-like structure or we can model as:
        # For each conflict sum s with pairs (i1,j1), (i2,j2), ..., (ik,jk):
        #   We need to remove at least one element from each pair.
        #   But removing one element can break multiple pairs.
        #
        # This is actually a minimum vertex cover in a hypergraph, which is NP-hard.
        # For small conflict sets, we can use greedy or ILP-like approach.
        
        # Greedy approach: repeatedly pick the element that appears in the most conflicts
        element_conflict_count = defaultdict(int)
        for s, pairs in conflicts:
            for i, j in pairs:
                element_conflict_count[i] += 1
                element_conflict_count[j] += 1
        
        # Convert to element values
        elem_vals = list(a)
        seed_idx = set(i for i, x in enumerate(elem_vals) if x in seed_set)
        
        removable_conflicts = {}
        for i, count in element_conflict_count.items():
            if i not in seed_idx:
                removable_conflicts[i] = count
        
        if not removable_conflicts:
            return []
        
        # Greedy: remove the element with highest conflict count
        to_remove = set()
        remaining_conflicts = list(conflicts)
        
        while remaining_conflicts and to_remove:
            # Recount
            element_conflict_count.clear()
            for s, pairs in remaining_conflicts:
                for i, j in pairs:
                    if i not in to_remove and j not in to_remove:
                        element_conflict_count[i] += 1
                        element_conflict_count[j] += 1
            
            best_elem = -1
            best_count = 0
            for i, count in element_conflict_count.items():
                if i not in seed_idx and i not in to_remove and count > best_count:
                    best_count = count
                    best_elem = i
            
            if best_elem == -1:
                break
            to_remove.add(best_elem)
        
        return [a[i] for i in to_remove]

    def local_search_bitset(a, sums_bits, max_iters=1000, deadline=None):
        """Local search: remove one, add one, then greedy extend."""
        a = list(a)
        sums_bits = sums_bits
        best_a = list(a)
        best_bits = sums_bits
        best_size = len(a)
        seed_set = set(seed_elements)
        
        candidates = [x for x in range(1, max_val + 1) if x not in a]
        
        for _ in range(max_iters):
            if deadline and time.time() > deadline:
                break
            
            if len(a) <= len(seed_elements):
                break
            
            removable = [x for x in a if x not in seed_set]
            if not removable:
                break
                
            rem_idx = random.randint(0, len(removable) - 1)
            x_remove = removable[rem_idx]
            
            a.remove(x_remove)
            new_sums_bits = compute_sums_bitset(a)
            
            improved = False
            for _ in range(10):
                if not candidates:
                    break
                x_add = random.choice(candidates)
                if x_add in a:
                    continue
                if can_add(x_add, a, new_sums_bits):
                    a.append(x_add)
                    a.sort()
                    new_sums_bits |= 1 << (2 * x_add)
                    for y in a:
                        if y != x_add:
                            new_sums_bits |= 1 << (x_add + y)
                    
                    a, new_sums_bits = greedy_extend_bitset(a, new_sums_bits)
                    
                    if len(a) > best_size:
                        best_a = list(a)
                        best_bits = new_sums_bits
                        best_size = len(a)
                    improved = True
                    break
            
            if not improved:
                a.append(x_remove)
                a.sort()
                new_sums_bits = compute_sums_bitset(a)
                
        return best_a, best_bits

    def swap_local_search(a, sums_bits, deadline=None):
        """Try all single-element swaps systematically."""
        a = list(a)
        sums_bits = sums_bits
        best_a = list(a)
        best_bits = sums_bits
        best_size = len(a)
        seed_set = set(seed_elements)
        
        removable = [x for x in a if x not in seed_set]
        candidates = [x for x in range(1, max_val + 1) if x not in a]
        
        # Try each removal
        for x_remove in removable:
            if deadline and time.time() > deadline:
                break
            
            a_tmp = [x for x in a if x != x_remove]
            bits_tmp = compute_sums_bitset(a_tmp)
            
            # Try to add each candidate
            for x_add in candidates:
                if deadline and time.time() > deadline:
                    break
                if can_add(x_add, a_tmp, bits_tmp):
                    a_new = a_tmp + [x_add]
                    a_new.sort()
                    bits_new = bits_tmp | (1 << (2 * x_add))
                    for y in a_new:
                        if y != x_add:
                            bits_new |= 1 << (x_add + y)
                    
                    # Greedy extend
                    a_ext, bits_ext = greedy_extend_bitset(a_new, bits_new)
                    
                    if len(a_ext) > best_size:
                        best_a = a_ext
                        best_bits = bits_ext
                        best_size = len(a_ext)
        
        return best_a, best_bits

    def conflict_polish(a, deadline=None):
        """Use min-cut-like greedy to resolve conflicts, then re-extend."""
        a = list(a)
        seed_set = set(seed_elements)
        
        # Check for conflicts
        sums_dict = compute_sums_dict(a)
        has_conflict = any(len(pairs) > 1 for pairs in sums_dict.values())
        
        if not has_conflict:
            return a
        
        # Greedy conflict resolution
        to_remove = min_cut_resolve(a, sums_dict, seed_set)
        
        if to_remove:
            a_new = [x for x in a if x not in to_remove]
            a_new.sort()
            bits = compute_sums_bitset(a_new)
            a_new, bits = greedy_extend_bitset(a_new, bits)
            return a_new
        
        return a

    # Initial greedy
    best_a = list(seed_elements)
    best_bits = compute_sums_bitset(best_a)
    best_a, best_bits = greedy_extend_bitset(best_a, best_bits)
    
    random.seed(42)
    
    # Multiple restarts with local search
    num_restarts = 100
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Randomized greedy start
        a = list(seed_elements)
        bits = compute_sums_bitset(a)
        
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if x in a:
                continue
            if can_add(x, a, bits):
                a.append(x)
                a.sort()
                bits |= 1 << (2 * x)
                for y in a:
                    if y != x:
                        bits |= 1 << (x + y)
                
        # Local search
        a_ls, bits_ls = local_search_bitset(a, bits, max_iters=200, deadline=deadline - 5)
        
        # Conflict polish
        a_pol = conflict_polish(a_ls, deadline=deadline - 5)
        if len(a_pol) > len(a_ls):
            a_ls = a_pol
            bits_ls = compute_sums_bitset(a_ls)
        
        # Swap local search
        if time.time() < deadline - 10:
            a_swap, bits_swap = swap_local_search(a_ls, bits_ls, deadline=deadline - 5)
            if len(a_swap) > len(a_ls):
                a_ls = a_swap
                bits_ls = bits_swap
        
        if len(a_ls) > len(best_a):
            best_a = a_ls
            best_bits = bits_ls
            
    # Final polish on best
    if time.time() < deadline - 5:
        best_a_pol = conflict_polish(best_a, deadline=deadline - 2)
        if len(best_a_pol) > len(best_a):
            best_a = best_a_pol
            best_bits = compute_sums_bitset(best_a)
        
        best_a_swap, best_bits_swap = swap_local_search(best_a, best_bits, deadline=deadline - 2)
        if len(best_a_swap) > len(best_a):
            best_a = best_a_swap
            best_bits = best_bits_swap
    
    # Final verification
    final_a = sorted(best_a)
    assert all(x in final_a for x in seed_elements), "Missing seed elements"
    assert all(1 <= x <= 284 for x in final_a), "Out of range"
    assert len(set(final_a)) == len(final_a), "Duplicates"
    
    # Verify Sidon property
    sums = set()
    n = len(final_a)
    valid = True
    for i in range(n):
        for j in range(i, n):
            s = final_a[i] + final_a[j]
            if s in sums:
                valid = False
                break
            sums.add(s)
        if not valid:
            break
    assert valid, "Invalid Sidon set"
    
    return final_a
# EVOLVE-BLOCK-END
