"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import heapq
import random
from itertools import permutations
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    if n <= 3:
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
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    # Standard recursive construction as baseline
    def standard_construction(n):
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
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word
    
    baseline_word = standard_construction(n)
    baseline_len = len(baseline_word)
    
    # For n=4, known shortest is 33
    if n == 4:
        best_word = None
        best_len = baseline_len
        
        symbols = tuple(range(1, n+1))
        all_perms = list(permutations(symbols))
        perm_to_idx = {p: i for i, p in enumerate(all_perms)}
        num_perms = len(all_perms)
        
        successor = [0] * num_perms
        for i, p in enumerate(all_perms):
            prefix = p[1:]
            used = set(prefix)
            missing = None
            for s in symbols:
                if s not in used:
                    missing = s
                    break
            q = prefix + (missing,)
            successor[i] = perm_to_idx[q]
        
        visited = [False] * num_perms
        cycle_len = 0
        for i in range(num_perms):
            if not visited[i]:
                j = i
                while not visited[j]:
                    visited[j] = True
                    cycle_len += 1
                    j = successor[j]
        
        if cycle_len == num_perms:
            path = [all_perms[0]]
            cur = 0
            for _ in range(num_perms - 1):
                cur = successor[cur]
                path.append(all_perms[cur])
            
            word = list(path[0])
            for p in path[1:]:
                word.append(p[-1])
            
            if len(word) < best_len:
                best_len = len(word)
                best_word = word
        
        if best_word:
            return "".join(map(str, best_word))
    
    # For n >= 5: Use shortest superstring formulation
    # Represent each permutation as a string, find shortest superstring containing all
    
    def compute_overlap(a, b, n):
        """Compute max overlap: suffix of a matches prefix of b, up to n-1"""
        max_ov = min(n - 1, len(a), len(b))
        for k in range(max_ov, -1, -1):
            if k == 0 or a[-k:] == b[:k]:
                return k
        return 0
    
    def greedy_eulerian_trail(perms_strs, n, time_limit):
        """
        Greedy Eulerian trail heuristic on the directed graph of length-n permutations.
        Nodes are permutations, edges represent valid suffix-prefix overlaps.
        Iteratively select next edge that minimizes increase in disconnected components.
        """
        m = len(perms_strs)
        if m <= 1:
            return perms_strs[0] if perms_strs else ""
        
        # Precompute overlap matrix
        ov = [[0] * m for _ in range(m)]
        for i in range(m):
            for j in range(m):
                if i != j:
                    ov[i][j] = compute_overlap(perms_strs[i], perms_strs[j], n)
        
        # Build adjacency: for each node, list of (neighbor, overlap)
        adj = [[] for _ in range(m)]
        for i in range(m):
            for j in range(m):
                if i != j and ov[i][j] > 0:
                    adj[i].append((j, ov[i][j]))
        
        # Track used edges
        used_edges = [[False] * m for _ in range(m)]
        
        # Compute initial connected components of unvisited edges
        def count_components():
            visited_nodes = [False] * m
            comps = 0
            for i in range(m):
                if not visited_nodes[i]:
                    # Check if node i has any unvisited incident edges
                    has_edge = False
                    for j in range(m):
                        if i != j and not used_edges[i][j]:
                            has_edge = True
                            break
                        if i != j and not used_edges[j][i]:
                            has_edge = True
                            break
                    if has_edge:
                        comps += 1
                        # BFS/DFS to mark all reachable nodes with unvisited edges
                        stack = [i]
                        visited_nodes[i] = True
                        while stack:
                            u = stack.pop()
                            for v in range(m):
                                if u != v and not visited_nodes[v]:
                                    if not used_edges[u][v] or not used_edges[v][u]:
                                        visited_nodes[v] = True
                                        stack.append(v)
            return comps
        
        initial_comps = count_components()
        
        # Start with node 0
        current = 0
        order = [0]
        
        deadline = start_time + time_limit
        
        while len(order) < m:
            if time.time() >= deadline:
                break
            
            best_next = -1
            best_score = float('inf')
            
            for next_node, overlap in adj[current]:
                if used_edges[current][next_node]:
                    continue
                
                # Tentatively mark this edge as used
                used_edges[current][next_node] = True
                
                # Count components after this move
                new_comps = count_components()
                
                # Score: prefer moves that keep components low
                # Also prefer higher overlap
                score = (new_comps - initial_comps) * 1000 - overlap
                
                if score < best_score:
                    best_score = score
                    best_next = next_node
                
                # Undo the tentative mark
                used_edges[current][next_node] = False
            
            if best_next == -1:
                # No valid moves, pick any unvisited node
                for i in range(m):
                    if i not in set(order):
                        best_next = i
                        break
                if best_next == -1:
                    break
            
            # Mark edge as used
            used_edges[current][best_next] = True
            current = best_next
            order.append(current)
        
        # Build superstring from order
        result = list(perms_strs[order[0]])
        for i in range(1, len(order)):
            k = ov[order[i-1]][order[i]]
            result.extend(perms_strs[order[i]][k:])
        return "".join(map(str, result))
    
    def local_search_polish(perms_strs, order, ov, n, time_limit):
        """
        Local search using 2-opt moves to improve the ordering.
        """
        m = len(perms_strs)
        if m <= 2:
            return order
        
        current_order = list(order)
        
        def compute_len(ord):
            total = len(perms_strs[ord[0]])
            for i in range(1, len(ord)):
                total += len(perms_strs[ord[i]]) - ov[ord[i-1]][ord[i]]
            return total
        
        current_len = compute_len(current_order)
        best_order = list(current_order)
        best_len = current_len
        
        deadline = start_time + time_limit
        no_improve_count = 0
        max_no_improve = 1000
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            improved = False
            for i in range(m - 1):
                if time.time() >= deadline:
                    break
                for j in range(i + 1, m):
                    if time.time() >= deadline:
                        break
                    
                    old_delta = 0
                    new_delta = 0
                    
                    if i > 0:
                        old_delta += ov[current_order[i-1]][current_order[i]]
                        new_delta += ov[current_order[i-1]][current_order[j]]
                    
                    if j < m - 1:
                        old_delta += ov[current_order[j]][current_order[j+1]]
                        new_delta += ov[current_order[i]][current_order[j+1]]
                    
                    for k in range(i, j):
                        old_delta += ov[current_order[k]][current_order[k+1]]
                        new_delta += ov[current_order[k+1]][current_order[k]]
                    
                    delta = new_delta - old_delta
                    if delta < 0:
                        current_order[i:j+1] = reversed(current_order[i:j+1])
                        current_len += delta
                        improved = True
                        if current_len < best_len:
                            best_len = current_len
                            best_order = list(current_order)
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                        break
                if improved:
                    break
            
            if not improved:
                no_improve_count += m * (m - 1) // 2
        
        return best_order
    
    def build_superstring(perms_strs, order, ov):
        """Build the final superstring from an ordering"""
        if not order:
            return ""
        result = list(perms_strs[order[0]])
        for i in range(1, len(order)):
            k = ov[order[i-1]][order[i]]
            result.extend(perms_strs[order[i]][k:])
        return "".join(map(str, result))
    
    # Get all permutations as strings
    symbols = list(range(1, n + 1))
    all_perms = list(permutations(symbols))
    perms_strs = ["".join(map(str, p)) for p in all_perms]
    
    # Time budget
    remaining = deadline - time.time()
    
    if remaining > 5:
        try:
            # Use greedy Eulerian trail heuristic
            improved_word = greedy_eulerian_trail(perms_strs, n, min(remaining - 2, 60))
            if len(improved_word) < baseline_len:
                baseline_word = list(map(int, improved_word))
                baseline_len = len(improved_word)
        except Exception:
            pass
        
        # Polish with local search if time remains
        remaining = deadline - time.time()
        if remaining > 10 and baseline_len < float('inf'):
            try:
                # Reconstruct order from baseline_word for local search
                # For simplicity, just run local search on a fresh greedy order
                m = len(perms_strs)
                ov = [[0] * m for _ in range(m)]
                for i in range(m):
                    for j in range(m):
                        if i != j:
                            ov[i][j] = compute_overlap(perms_strs[i], perms_strs[j], n)
                
                # Start with identity order
                init_order = list(range(m))
                polished_order = local_search_polish(perms_strs, init_order, ov, n, min(remaining - 2, 30))
                polished_word = build_superstring(perms_strs, polished_order, ov)
                if len(polished_word) < baseline_len:
                    baseline_word = list(map(int, polished_word))
                    baseline_len = len(polished_word)
            except Exception:
                pass
    
    return "".join(map(str, baseline_word))
# EVOLVE-BLOCK-END
