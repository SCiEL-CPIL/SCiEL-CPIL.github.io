import time
import random

N = 17
TOTAL_INPUTS = 1 << N  # 131072


def _build_initial_masks():
    masks = [0] * N
    for m in range(TOTAL_INPUTS):
        for k in range(N):
            if (m >> k) & 1:
                masks[k] |= (1 << m)
    return masks


def _simulate_tail(masks_before, tail_comparators):
    masks = masks_before[:]
    for i, j in tail_comparators:
        mi = masks[i]
        mj = masks[j]
        masks[i] = mi & mj
        masks[j] = mi | mj
    return masks


def _is_sorted(masks):
    for k in range(N - 1):
        if (masks[k] & ~masks[k + 1]) != 0:
            return False
    return True


def _build_states(network, initial_masks):
    states = [initial_masks]
    current_masks = initial_masks
    for comp in network:
        i, j = comp
        mi = current_masks[i]
        mj = current_masks[j]
        new_masks = current_masks[:]
        new_masks[i] = mi & mj
        new_masks[j] = mi | mj
        current_masks = new_masks
        states.append(current_masks)
    return states


def _greedy_delete(current_net, initial_masks):
    if not current_net:
        return current_net
    states = _build_states(current_net, initial_masks)
    if not _is_sorted(states[-1]):
        return current_net
    changed = True
    while changed:
        changed = False
        for k in range(len(current_net) - 1, -1, -1):
            if k == len(current_net) - 1:
                if _is_sorted(states[k]):
                    current_net.pop(k)
                    states.pop()
                    changed = True
                    break
            else:
                tail = current_net[k + 1:]
                final_masks = _simulate_tail(states[k], tail)
                if _is_sorted(final_masks):
                    current_net.pop(k)
                    states = _build_states(current_net, initial_masks)
                    changed = True
                    break
    return current_net


def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # 1. Generate initial network: Odd-Even Transposition
    network = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            network.append((i, i + 1))

    # 2. Pre-compute initial masks
    initial_masks = _build_initial_masks()

    # 3. Greedy deletion on seed
    network = _greedy_delete(network, initial_masks)
    best_network = list(network)
    best_len = len(network)

    # 4. Local search
    search_deadline = deadline - 3.0
    for _ in range(2000):
        if time.time() >= search_deadline:
            break

        current_net = list(best_network)
        move_type = random.randint(0, 2)

        if move_type == 0:
            pos = random.randint(0, len(current_net))
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            current_net.insert(pos, (i, j))
        elif move_type == 1 and len(current_net) > 0:
            pos = random.randint(0, len(current_net) - 1)
            current_net.pop(pos)
        elif move_type == 2 and len(current_net) >= 2:
            k1 = random.randint(0, len(current_net) - 1)
            k2 = random.randint(0, len(current_net) - 1)
            if k1 != k2:
                current_net[k1], current_net[k2] = current_net[k2], current_net[k1]

        current_net = _greedy_delete(current_net, initial_masks)

        if len(current_net) < best_len:
            best_network = current_net
            best_len = len(current_net)

    return best_network
