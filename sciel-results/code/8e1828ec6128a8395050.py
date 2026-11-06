import time
import random
import numpy as np

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 55  # 5s buffer

    # Pre-compute all 4-APs
    # AP is (a, b, c, d) such that b-a = c-b = d-c = diff
    aps = []
    element_to_aps = [[] for _ in range(N)]

    for d in range(1, N):
        max_a = N - 3 * d
        for a in range(0, max_a):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap_idx = len(aps)
            aps.append((a, b, c, e))
            element_to_aps[a].append(ap_idx)
            element_to_aps[b].append(ap_idx)
            element_to_aps[c].append(ap_idx)
            element_to_aps[e].append(ap_idx)

    num_aps = len(aps)

    # Helper functions for state management
    # in_set: numpy bool array
    # ap_count: numpy int array, count of elements from AP currently in set

    def init_state():
        in_set = np.zeros(N, dtype=np.bool_)
        ap_count = np.zeros(num_aps, dtype=np.int32)
        return in_set, ap_count

    def add_element(x, in_set, ap_count):
        # Check if adding x creates a 4-AP
        for ap_idx in element_to_aps[x]:
            if ap_count[ap_idx] == 3:
                return False
        # Safe to add
        in_set[x] = True
        for ap_idx in element_to_aps[x]:
            ap_count[ap_idx] += 1
        return True

    def remove_element(x, in_set, ap_count):
        in_set[x] = False
        for ap_idx in element_to_aps[x]:
            ap_count[ap_idx] -= 1

    def get_size(in_set):
        return np.sum(in_set)

    # 1. Initial Greedy Pass to get a good starting point
    # We will run multiple greedy restarts to find the best initial set
    best_in_set = None
    best_ap_count = None
    best_size = 0

    # Run a few greedy passes
    for _ in range(30):
        in_set, ap_count = init_state()
        candidates = list(range(N))
        random.shuffle(candidates)
        for x in candidates:
            if add_element(x, in_set, ap_count):
                pass

        size = get_size(in_set)
        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()
            best_ap_count = ap_count.copy()

    # 2. Local Search with Kicks and Pair Swaps
    current_in_set = best_in_set
    current_ap_count = best_ap_count
    current_size = best_size

    # Kick parameters
    kick_sizes = [1, 2, 3]

    no_improve_count = 0
    max_no_improve = 200

    while time.time() < deadline:
        # Try to add any available element (greedy improvement)
        improved = False
        for x in range(N):
            if not current_in_set[x]:
                if add_element(x, current_in_set, current_ap_count):
                    current_size += 1
                    improved = True
        if improved:
            no_improve_count = 0
            continue

        no_improve_count += 1
        if no_improve_count > max_no_improve:
            no_improve_count = 0
            max_no_improve = min(max_no_improve + 50, 1000)

        # Perform a kick
        # Choose a random kick size
        k = random.choice(kick_sizes)

        # Get list of elements in set
        current_elements = np.where(current_in_set)[0]

        if len(current_elements) < k:
            # Set is too small, just add random elements if possible
            for x in range(N):
                if not current_in_set[x]:
                    if add_element(x, current_in_set, current_ap_count):
                        pass
            continue

        # Select k random elements to remove
        removed_elements = random.sample(current_elements.tolist(), k)

        # Store state before kick to revert if bad
        old_in_set = current_in_set.copy()
        old_ap_count = current_ap_count.copy()

        # Remove elements
        for x in removed_elements:
            remove_element(x, current_in_set, current_ap_count)

        # Try to add back elements
        # Iterate through all elements not in set and try to add them
        for x in range(N):
            if not current_in_set[x]:
                if add_element(x, current_in_set, current_ap_count):
                    pass

        new_size = get_size(current_in_set)

        # Acceptance
        if new_size >= current_size:
            # Accept (including same size to explore)
            current_size = new_size
            no_improve_count = 0
        else:
            # Revert
            current_in_set = old_in_set
            current_ap_count = old_ap_count

        # Check for completion
        if current_size > 250:
            break

    # 3. Branch-and-Bound Polish: Try to find if we can add more elements
    # by removing a small set of elements
    # We'll try all pairs of elements to remove and see if we can add 3+ back
    polish_deadline = start_time + 53  # 2s before hard deadline

    while time.time() < polish_deadline:
        current_elements = np.where(current_in_set)[0]
        if len(current_elements) < 2:
            break

        # Try random pair swaps: remove 2, add as many as possible
        for _ in range(200):
            if time.time() >= polish_deadline:
                break

            i, j = random.sample(range(len(current_elements)), 2)
            x1, x2 = int(current_elements[i]), int(current_elements[j])

            # Store state
            old_in_set = current_in_set.copy()
            old_ap_count = current_ap_count.copy()

            # Remove x1, x2
            remove_element(x1, current_in_set, current_ap_count)
            remove_element(x2, current_in_set, current_ap_count)

            # Try to add all possible elements
            added = 0
            for x in range(N):
                if not current_in_set[x]:
                    if add_element(x, current_in_set, current_ap_count):
                        added += 1

            new_size = current_size - 2 + added

            if new_size > current_size:
                current_size = new_size
            else:
                # Revert
                current_in_set = old_in_set
                current_ap_count = old_ap_count

    # Final polish: try to add any remaining elements
    for x in range(N):
        if not current_in_set[x]:
            if add_element(x, current_in_set, current_ap_count):
                pass

    # Convert to 1-indexed list
    result = [i + 1 for i in range(N) if current_in_set[i]]
    return sorted(result)
