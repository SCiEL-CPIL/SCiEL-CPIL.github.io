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
    
    # For n >= 5: Use shortest superstring formulation with branch-and-bound
    # Represent each permutation as a string, find shortest superstring containing all
    def compute_overlap(a, b, n):
        """Compute max overlap: suffix of a matches prefix of b, up to n-1"""
        max_ov = min(n - 1, len(a), len(b))
        for k in range(max_ov, -1, -1):
            if k == 0 or a[-k:] == b[:k]:
                return k
        return 0
    
    def shortest_superstring_bnb(perms_strs, n, time_limit):
        """
        Branch and bound to find shortest superstring containing all permutations.
        Uses bitmask DP with pruning.
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
        
        # Lower bound: use maximum spanning tree of overlap graph
        # For pruning, we'll use a simple heuristic lower bound
        
        best_len = float('inf')
        best_order = None
        
        # State: (current_perm, visited_mask, current_length, order_so_far)
        # We'll use iterative deepening with branch and bound
        
        # First, get a good initial solution with greedy
        greedy_order = greedy_ordering(perms_strs, ov, n)
        greedy_len = compute_total_length(perms_strs, greedy_order, ov)
        best_len = greedy_len
        best_order = greedy_order
        
        # Now try to improve with local search (2-opt style)
        improved_order = local_search_2opt(perms_strs, ov, n, best_order, best_len, time_limit)
        if improved_order:
            new_len = compute_total_length(perms_strs, improved_order, ov)
            if new_len < best_len:
                best_len = new_len
                best_order = improved_order
        
        # Build final string
        return build_superstring(perms_strs, best_order, ov)
    
    def greedy_ordering(perms_strs, ov, n):
        """Greedy nearest-neighbor for initial solution"""
        m = len(perms_strs)
        if m <= 1:
            return list(range(m))
        
        # Start with the string that has best average outgoing overlap
        start = 0
        best_avg = -1
        for i in range(m):
            avg = sum(ov[i][j] for j in range(m) if j != i) / (m - 1)
            if avg > best_avg:
                best_avg = avg
                start = i
        
        unvisited = set(range(m))
        unvisited.remove(start)
        order = [start]
        current = start
        
        while unvisited:
            best_next = -1
            best_ov = -1
            for j in unvisited:
                if ov[current][j] > best_ov:
                    best_ov = ov[current][j]
                    best_next = j
            if best_next == -1:
                # Pick any remaining
                best_next = next(iter(unvisited))
            current = best_next
            unvisited.remove(current)
            order.append(current)
        
        return order
    
    def compute_total_length(perms_strs, order, ov):
        """Compute total length of superstring given an ordering"""
        if not order:
            return 0
        total = len(perms_strs[order[0]])
        for i in range(1, len(order)):
            total += len(perms_strs[order[i]]) - ov[order[i-1]][order[i]]
        return total
    
    def local_search_2opt(perms_strs, ov, n, initial_order, initial_len, time_limit):
        """
        Local search using 2-opt moves to improve the ordering.
        """
        m = len(perms_strs)
        if m <= 2:
            return initial_order
        
        current_order = list(initial_order)
        current_len = initial_len
        best_order = list(current_order)
        best_len = current_len
        
        deadline = start_time + time_limit
        no_improve_count = 0
        max_no_improve = 2000
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            improved = False
            # Try all 2-opt moves
            for i in range(m - 1):
                if time.time() >= deadline:
                    break
                for j in range(i + 1, m):
                    if time.time() >= deadline:
                        break
                    
                    # Compute new length with segment [i..j] reversed
                    # Old contributions: ov[order[i-1]][order[i]] + ov[order[j]][order[j+1]] (if exist)
                    # New contributions: ov[order[i-1]][order[j]] + ov[order[i]][order[j+1]] (if exist)
                    
                    old_delta = 0
                    new_delta = 0
                    
                    # Left boundary
                    if i > 0:
                        old_delta += ov[current_order[i-1]][current_order[i]]
                        new_delta += ov[current_order[i-1]][current_order[j]]
                    
                    # Right boundary
                    if j < m - 1:
                        old_delta += ov[current_order[j]][current_order[j+1]]
                        new_delta += ov[current_order[i]][current_order[j+1]]
                    
                    # Internal edges in reversed segment
                    # Old: sum of ov[order[k]][order[k+1]] for k in [i, j-1]
                    # New: sum of ov[order[k+1]][order[k]] for k in [i, j-1] (reversed)
                    for k in range(i, j):
                        old_delta += ov[current_order[k]][current_order[k+1]]
                        new_delta += ov[current_order[k+1]][current_order[k]]
                    
                    delta = new_delta - old_delta
                    if delta < 0:
                        # Apply the 2-opt move
                        current_order[i:j+1] = reversed(current_order[i:j+1])
                        current_len += delta
                        improved = True
                        if current_len < best_len:
                            best_len = current_len
                            best_order = list(current_order)
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                    
                    if improved:
                        # Restart the scan from beginning after improvement
                        break
                
                if improved:
                    break
            
            if not improved:
                no_improve_count += m * (m - 1) // 2
        
        return best_order if best_len < initial_len else initial_order
    
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
    
    # Time budget for the BnB search
    elapsed = time.time() - start_time
    remaining = deadline - time.time()
    
    if remaining > 5:
        # Use the shortest superstring approach
        try:
            improved_word = shortest_superstring_bnb(perms_strs, n, min(remaining - 2, 90))
            if len(improved_word) < baseline_len:
                baseline_word = list(map(int, improved_word))
                baseline_len = len(improved_word)
        except Exception:
            pass
    
    return "".join(map(str, baseline_word))
# EVOLVE-BLOCK-END
