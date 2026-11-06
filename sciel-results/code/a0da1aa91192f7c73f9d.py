import time
import math
import random

def construct():
    start_time = time.time()
    # Safety margin: 90s limit, use 80s for computation
    deadline = start_time + 80.0

    N = 256

    # 1. Preprocessing: Identify invalid nodes and build conflict graph
    squares = set()
    k = 2
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Nodes where 2*a is a perfect square are individually invalid
    individually_invalid = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            individually_invalid.add(a)

    valid_nodes = [i for i in range(1, N + 1) if i not in individually_invalid]
    valid_set = set(valid_nodes)
    n = len(valid_nodes)
    node_to_idx = {node: i for i, node in enumerate(valid_nodes)}

    # Build conflict bitsets for valid nodes
    # conflict_bits[i] is a bitmask of nodes j (0 <= j < n) such that valid_nodes[i] + valid_nodes[j] is a square
    conflict_bits = [0] * n
    for i, node in enumerate(valid_nodes):
        for s in squares:
            j_val = s - node
            if 1 <= j_val <= N and j_val in valid_set:
                j_idx = node_to_idx[j_val]
                conflict_bits[i] |= (1 << j_idx)
        # Ensure self is not in conflict (2*node is not square for valid nodes, but just in case)
        conflict_bits[i] &= ~(1 << i)

    def popcount(x):
        return bin(x).count('1')

    def greedy_min_degree():
        """Greedy min-degree independent set."""
        remaining = (1 << n) - 1
        selected_bits = 0
        while remaining:
            best_node = -1
            best_degree = n + 1
            for i in range(n):
                if (remaining >> i) & 1:
                    deg = popcount(conflict_bits[i] & remaining)
                    if deg < best_degree:
                        best_degree = deg
                        best_node = i
            if best_node == -1:
                break
            selected_bits |= (1 << best_node)
            remaining &= ~(1 << best_node)
            remaining &= ~conflict_bits[best_node]
        return selected_bits

    def greedy_add_pass(bits, size, time_limit):
        """Repeatedly add any non-conflicting node until no more can be added."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(n):
                if (bits >> i) & 1:
                    continue
                if conflict_bits[i] & bits:
                    continue
                bits |= (1 << i)
                size += 1
                improved = True
        return bits, size

    def greedy_is_size(candidates_bits):
        """
        Estimate the size of the maximum independent set in the subgraph
        defined by candidates_bits using a greedy heuristic.
        This serves as an upper bound for B&B.
        """
        rem = candidates_bits
        count = 0
        while rem:
            # Pick a vertex with minimum degree in the remaining subgraph
            best_node = -1
            best_deg = n + 1
            # Optimization: Iterate only over set bits
            temp = rem
            while temp:
                lsb = temp & -temp
                i = lsb.bit_length() - 1
                deg = popcount(conflict_bits[i] & rem)
                if deg < best_deg:
                    best_deg = deg
                    best_node = i
                temp ^= lsb
            
            if best_node == -1:
                break
            count += 1
            rem &= ~(1 << best_node)
            rem &= ~conflict_bits[best_node]
        return count

    def branch_and_bound(current_bits, candidates_bits, current_size, best_size_holder):
        """
        Recursive B&B to find MIS.
        best_size_holder is a list [size, mask] to allow modification in recursion.
        """
        if time.time() > deadline:
            return

        # Upper bound check
        # If current_size + greedy_bound <= best_size, we can prune
        bound = greedy_is_size(candidates_bits)
        if current_size + bound <= best_size_holder[0]:
            return

        if candidates_bits == 0:
            if current_size > best_size_holder[0]:
                best_size_holder[0] = current_size
                best_size_holder[1] = current_bits
            return

        # Select a branching vertex
        # Heuristic: Pick a vertex with high degree in the candidate subgraph
        # to reduce the search space quickly
        best_node = -1
        best_deg = -1
        temp = candidates_bits
        while temp:
            lsb = temp & -temp
            i = lsb.bit_length() - 1
            deg = popcount(conflict_bits[i] & candidates_bits)
            if deg > best_deg:
                best_deg = deg
                best_node = i
            temp ^= lsb

        if best_node == -1:
            return

        # Branch 1: Include best_node
        new_current = current_bits | (1 << best_node)
        # Remove best_node and its neighbors from candidates
        new_candidates = candidates_bits & ~(1 << best_node) & ~conflict_bits[best_node]
        branch_and_bound(new_current, new_candidates, current_size + 1, best_size_holder)

        if time.time() > deadline:
            return

        # Branch 2: Exclude best_node
        new_candidates_excl = candidates_bits & ~(1 << best_node)
        branch_and_bound(current_bits, new_candidates_excl, current_size, best_size_holder)

    # --- Main Execution Flow ---

    # Phase 1: Initial Solution via Greedy Min-Degree
    random.seed(42)
    best_mask = greedy_min_degree()
    best_size = popcount(best_mask)
    
    # Phase 2: Polish with Greedy Add
    best_mask, best_size = greedy_add_pass(best_mask, best_size, deadline - 60.0)

    # Phase 3: Local Search (Swap) to improve initial solution
    # This helps provide a tighter lower bound for B&B
    unselected = [i for i in range(n) if not ((best_mask >> i) & 1)]
    improved = True
    while improved and time.time() < deadline - 40.0:
        improved = False
        for a in unselected:
            if (best_mask >> a) & 1:
                continue
            conflicting = [b for b in range(n) if (best_mask >> b) & 1 and (conflict_bits[a] >> b) & 1]
            if len(conflicting) <= 1:
                continue
            removed_bits = 0
            for b in conflicting:
                removed_bits |= (1 << b)
            candidate_bits = (best_mask & ~removed_bits) | (1 << a)
            temp_bits = candidate_bits
            temp_size = popcount(candidate_bits)
            # Try to add more nodes
            for c in unselected:
                if c == a or (removed_bits >> c) & 1:
                    continue
                if (temp_bits >> c) & 1:
                    continue
                if conflict_bits[c] & temp_bits:
                    continue
                temp_bits |= (1 << c)
                temp_size += 1
            if temp_size > best_size:
                best_mask = temp_bits
                best_size = temp_size
                unselected = [i for i in range(n) if not ((best_mask >> i) & 1)]
                improved = True
                break

    # Phase 4: Branch and Bound
    # We run B&B to try to find a solution larger than best_size
    # We use a holder for the best result found by B&B
    best_holder = [best_size, best_mask]
    
    # Initial candidates: All valid nodes not in the current best set?
    # Actually, B&B should start from scratch or from a relaxed state.
    # Starting from scratch with a good initial bound is standard.
    # However, the search space is 2^n.
    # To make B&B effective, we should perhaps start with all nodes as candidates.
    
    if time.time() < deadline - 5.0:
        initial_candidates = (1 << n) - 1
        branch_and_bound(0, initial_candidates, 0, best_holder)

    # Final Polish
    if time.time() < deadline - 2.0:
        final_mask, final_size = greedy_add_pass(best_holder[1], best_holder[0], deadline - 1.0)
        if final_size > best_holder[0]:
            best_holder[0] = final_size
            best_holder[1] = final_mask

    # Extract result
    result = []
    for i in range(n):
        if (best_holder[1] >> i) & 1:
            result.append(valid_nodes[i])

    result.sort()

    # Verify
    valid = True
    for i in range(len(result)):
        for j in range(i, len(result)):
            s = result[i] + result[j]
            r = math.isqrt(s)
            if r * r == s:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback
        result = []
        for i in range(n):
            if (greedy_min_degree() >> i) & 1:
                result.append(valid_nodes[i])
        result.sort()

    if not result:
        return [1]

    return result
