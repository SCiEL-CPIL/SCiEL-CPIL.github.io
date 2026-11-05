import time
import random
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin

    N = 1000
    best_set = None
    best_size = 0

    def can_add(x, chosen_set, counts):
        if counts.get(2 * x, 0) >= 2:
            return False
        for y in chosen_set:
            if y == x:
                continue
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
        return True

    def add_element(x, chosen_set, counts):
        chosen_set.add(x)
        s = 2 * x
        counts[s] = counts.get(s, 0) + 1
        for y in chosen_set:
            if y == x:
                continue
            s = x + y
            counts[s] = counts.get(s, 0) + 1

    def remove_element(x, chosen_set, counts):
        chosen_set.remove(x)
        s = 2 * x
        counts[s] -= 1
        if counts[s] == 0:
            del counts[s]
        for y in chosen_set:
            s = x + y
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]

    def conflict_cost(x, chosen_set, counts):
        """Compute the future conflict cost: number of sums that are saturated or close to saturation."""
        cost = 0
        if counts.get(2 * x, 0) >= 1:
            cost += 1
        for y in chosen_set:
            if y == x:
                continue
            s = x + y
            if counts.get(s, 0) >= 1:
                cost += 1
        return cost

    def priority_greedy():
        """Greedy using priority queue: always pick element with minimum conflict cost."""
        chosen_set = set()
        counts = {}
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)

        # Start with all candidates in the priority queue
        # We'll use lazy deletion: mark processed elements and skip them when popped
        pq = []
        for i, x in enumerate(candidates):
            heapq.heappush(pq, (0, i, x))  # initial cost 0

        processed = set()
        cost_cache = {}  # x -> last computed cost

        while pq and time.time() < deadline:
            cost, _, x = heapq.heappop(pq)
            if x in processed:
                continue
            if x in chosen_set:
                continue

            # Recompute cost to check if it's still valid
            current_cost = conflict_cost(x, chosen_set, counts)
            if current_cost != cost:
                # Cost changed, reinsert with new cost
                heapq.heappush(pq, (current_cost, random.random(), x))
                continue

            # Check if we can add x
            if can_add(x, chosen_set, counts):
                add_element(x, chosen_set, counts)
                processed.add(x)
                cost_cache[x] = current_cost

        return chosen_set, counts

    def greedy_construction():
        chosen_set = set()
        counts = {}
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)

        for x in candidates:
            if time.time() > deadline:
                break
            if can_add(x, chosen_set, counts):
                add_element(x, chosen_set, counts)

        return chosen_set, counts

    def local_search(chosen_set, counts):
        chosen_set = set(chosen_set)
        counts = dict(counts)
        unchosen = set(range(1, N + 1)) - chosen_set

        improved = True
        passes = 0
        max_passes = 10

        while improved and passes < max_passes and time.time() < deadline:
            improved = False
            passes += 1

            unchosen_list = list(unchosen)
            random.shuffle(unchosen_list)

            for x in unchosen_list:
                if time.time() > deadline:
                    break

                if can_add(x, chosen_set, counts):
                    add_element(x, chosen_set, counts)
                    unchosen.remove(x)
                    improved = True
                    continue

                if not chosen_set:
                    continue

                y = random.choice(list(chosen_set))

                remove_element(y, chosen_set, counts)

                if can_add(x, chosen_set, counts):
                    add_element(x, chosen_set, counts)
                    unchosen.remove(x)
                    unchosen.add(y)
                    improved = True
                else:
                    add_element(y, chosen_set, counts)

        return chosen_set, counts

    for restart in range(100):
        if time.time() > deadline:
            break

        # Alternate between different strategies
        if restart % 3 == 0:
            chosen_set, counts = priority_greedy()
            if time.time() < deadline:
                chosen_set, counts = local_search(chosen_set, counts)
        elif restart % 3 == 1:
            chosen_set, counts = greedy_construction()
            chosen_set, counts = local_search(chosen_set, counts)
        else:
            # Try a different randomization
            chosen_set, counts = greedy_construction()
            if time.time() < deadline:
                chosen_set, counts = local_search(chosen_set, counts)

        current_size = len(chosen_set)
        if current_size > best_size:
            best_size = current_size
            best_set = list(chosen_set)

    if best_set is None:
        best_set = []

    return best_set

random.seed(42)
