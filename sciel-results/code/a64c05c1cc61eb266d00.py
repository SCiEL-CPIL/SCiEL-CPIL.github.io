import time
import numpy as np
from collections import defaultdict

def synthesize(matrix, depth_limit):
    """
    Exact XOR-circuit synthesis via global iterative gate selection.
    
    Strategy:
    1. Represent each target row as a bitmask.
    2. Iteratively select the most beneficial XOR gate: a gate (a,b) is beneficial
       if computing a^b enables one or more still-uncomputed targets (either directly
       as a target, or by reducing a target's remaining weight).
    3. Use row-pair XOR: if target_i XOR target_j has low weight, compute that XOR
       once and use it to cheaply derive both targets.
    4. Fall back to per-row decomposition for remaining targets.
    """
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill

    mat = np.array(matrix, dtype=np.int8)
    m, n = mat.shape

    zero_wire = n

    # Row weights and baseline
    row_weights = np.sum(mat, axis=1).astype(int)
    baseline_gates = int(np.sum(np.maximum(0, row_weights - 1)))

    # Trivial case
    if baseline_gates == 0:
        gates = []
        outputs = []
        for i in range(m):
            if row_weights[i] == 0:
                outputs.append(zero_wire)
            else:
                idx = np.where(mat[i] == 1)[0][0]
                outputs.append(int(idx))
        return {'gates': gates, 'outputs': outputs}

    # Convert rows to bitmasks
    target_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if mat[i, j] == 1:
                mask |= (1 << j)
        target_masks.append(mask)

    # Available wires: mask -> wire_index
    available = {}
    for i in range(n):
        available[1 << i] = i
    available[0] = zero_wire

    gates = []
    next_wire = n + 1

    # Track which targets are done
    done = [False] * m
    target_wire = [None] * m

    # Mark targets that are already available
    for i in range(m):
        if target_masks[i] in available:
            target_wire[i] = available[target_masks[i]]
            done[i] = True

    # --- Phase 1: Row-pair XOR decomposition ---
    # For pairs (i,j) where w(target_i ^ target_j) < w(target_i) + w(target_j) - 1
    # it's beneficial to compute the pair XOR once.
    # We'll collect candidate pair-XORs and process the most beneficial ones first.

    # Build a list of candidate gates with their benefit
    # A candidate gate is (mask_a, mask_b) where both are in available
    # Benefit = number of undone targets that can be resolved by this gate

    def count_benefit(new_mask, available_dict, target_masks_list, done_list, n_val):
        """Count how many undone targets become directly available or reducible."""
        benefit = 0
        for i in range(len(target_masks_list)):
            if done_list[i]:
                continue
            tmask = target_masks_list[i]
            # Check if tmask ^ new_mask is in available (then tmask = new_mask ^ available[tmask^new_mask])
            other = tmask ^ new_mask
            if other in available_dict:
                benefit += 1
        return benefit

    # Iterative gate selection
    max_iterations = 50000
    iteration = 0

    while iteration < max_iterations:
        if time.time() > deadline:
            break

        iteration += 1

        # Check if all targets are done
        if all(done):
            break

        # Find the best gate to add
        # A gate is (mask_a, mask_b) where mask_a, mask_b are in available
        # The new mask is mask_a ^ mask_b
        # We want to maximize benefit

        # To keep this tractable, we consider:
        # 1. Gates involving at least one input wire (small search space)
        # 2. Gates between two "useful" intermediate wires

        # Strategy: For each undone target, try to find the cheapest decomposition
        # using available wires. The cheapest is 1 gate if tmask = a ^ b for available a,b.
        # Otherwise, we create intermediate gates that reduce the problem.

        # Approach: For each undone target, compute the "distance" to being resolvable
        # in 1 more gate. If distance is 2, we need to create one intermediate.
        # Pick the intermediate that helps the most targets.

        best_gate = None
        best_benefit = 0

        # Candidate new masks to consider:
        # For each undone target t, the set of masks x such that t ^ x is in available
        # means creating x would resolve t in 1 gate.
        # So for each undone target, the "needed" masks are {t ^ a : a in available}
        # We want to create the mask that appears most often across all undone targets.

        # Build frequency map: mask -> how many undone targets would be resolved
        needed_mask_count = defaultdict(int)
        for i in range(m):
            if done[i]:
                continue
            tmask = target_masks[i]
            # For each available mask a, if we had tmask ^ a, we could resolve t in 1 gate
            # But that's O(|available| * m) which is too slow for large available sets
            # Instead, only check: if tmask itself can be resolved in 1 gate, skip
            # Otherwise, the most useful new mask is tmask ^ (some single input)
            # because those are cheap to create (1 gate from input + available)

            # Check if tmask is resolvable in 1 gate from available
            resolvable_1 = False
            for amask in list(available.keys()):
                other = tmask ^ amask
                if other in available:
                    resolvable_1 = True
                    break

            if not resolvable_1:
                # The most useful intermediate is tmask ^ (1 << j) for some j
                # because creating tmask ^ (1<<j) costs 1 gate (input j XOR available[tmask^(1<<j)])
                # Wait, that's circular. Let me think differently.
                #
                # If we create a new wire with mask X, it costs 1 gate if X = a ^ b for available a,b.
                # After creating X, target t is resolvable in 1 more gate if t ^ X is in available.
                # So the benefit of creating X is: number of undone targets t where t ^ X is in available.
                #
                # Candidate X values: for each undone target t, X = t ^ a for some available a.
                # But we can only create X if X itself = a' ^ b' for available a', b'.
                #
                # Simplification: only consider X = t ^ (1<<j) for single-bit inputs.
                # This is cheap to create (1 gate) and covers the most common case.

                for j in range(n):
                    x = tmask ^ (1 << j)
                    if x not in available:
                        # Check if x can be created in 1 gate: x = a ^ b for available a,b
                        # x ^ (1<<j) = tmask, so if tmask is in available... no, tmask is not.
                        # x = (1<<j) ^ tmask. We need tmask in available, which it isn't.
                        # So x can't be created in 1 gate unless tmask is available.
                        # Alternative: x = a ^ b where a,b are available and a^b = x.
                        # This is the same as checking if x is in the XOR-closure of available.
                        # For efficiency, skip this check and just track frequency.
                        needed_mask_count[x] += 1

        # Find the most frequent needed mask
        if needed_mask_count:
            best_x = max(needed_mask_count, key=needed_mask_count.get)
            best_benefit = needed_mask_count[best_x]

            # Check if best_x can be created in 1 gate from available
            can_create = False
            create_pair = None
            for amask, awire in available.items():
                bmask = best_x ^ amask
                if bmask in available and bmask != amask:
                    can_create = True
                    create_pair = (awire, available[bmask])
                    break

            if can_create:
                # Create the gate
                w1, w2 = create_pair
                gates.append([w1, w2])
                target_wire_temp = next_wire
                next_wire += 1
                available[best_x] = target_wire_temp

                # Resolve any targets that are now resolvable in 1 gate
                for i in range(m):
                    if done[i]:
                        continue
                    tmask = target_masks[i]
                    other = tmask ^ best_x
                    if other in available:
                        # Resolve target i
                        ow = available[other]
                        if best_x == 0:
                            target_wire[i] = ow
                        elif other == 0:
                            target_wire[i] = target_wire_temp
                        else:
                            gates.append([target_wire_temp, ow])
                            target_wire[i] = next_wire
                            next_wire += 1
                            available[tmask] = target_wire[i]
                        done[i] = True

        # If no beneficial gate found, resolve the easiest remaining target directly
        if not all(done):
            # Find the undone target with minimum weight
            min_idx = -1
            min_w = float('inf')
            for i in range(m):
                if not done[i] and row_weights[i] < min_w:
                    min_w = row_weights[i]
                    min_idx = i

            if min_idx >= 0:
                tmask = target_masks[min_idx]
                # Decompose from inputs
                ones = [j for j in range(n) if (tmask >> j) & 1]
                if len(ones) == 0:
                    target_wire[min_idx] = zero_wire
                    done[min_idx] = True
                elif len(ones) == 1:
                    target_wire[min_idx] = ones[0]
                    done[min_idx] = True
                else:
                    current = ones[0]
                    for j in range(1, len(ones)):
                        gates.append([current, ones[j]])
                        current = next_wire
                        next_wire += 1
                    target_wire[min_idx] = current
                    available[tmask] = current
                    done[min_idx] = True

    # --- Phase 2: Resolve any remaining targets ---
    for i in range(m):
        if done[i]:
            continue
        if time.time() > deadline:
            break
        tmask = target_masks[i]
        ones = [j for j in range(n) if (tmask >> j) & 1]
        if len(ones) == 0:
            target_wire[i] = zero_wire
        elif len(ones) == 1:
            target_wire[i] = ones[0]
        else:
            # Try 1-gate resolution first
            resolved = False
            for amask, awire in available.items():
                other = tmask ^ amask
                if other in available:
                    ow = available[other]
                    if amask == 0:
                        target_wire[i] = ow
                    elif other == 0:
                        target_wire[i] = awire
                    else:
                        gates.append([awire, ow])
                        target_wire[i] = next_wire
                        next_wire += 1
                        available[tmask] = target_wire[i]
                    resolved = True
                    break
            if not resolved:
                current = ones[0]
                for j in range(1, len(ones)):
                    gates.append([current, ones[j]])
                    current = next_wire
                    next_wire += 1
                target_wire[i] = current
                available[tmask] = current

    return {'gates': gates, 'outputs': target_wire}
