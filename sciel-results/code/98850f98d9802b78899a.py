"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque

def pebble(dag, fast_memory_budget):
    t0 = time.time()
    deadline = 82.0  # safety margin before 90s kill

    def time_left():
        return deadline - (time.time() - t0)

    # Helper to validate a schedule
    def validate_schedule(ops, parents, outputs, budget, n):
        if len(ops) > 20000:
            return False
        red = set()
        stored_outputs = set()
        for op, v in ops:
            if op == 'load':
                if v in red:
                    return False
                if len(parents[v]) != 0:
                    return False
                if len(red) >= budget:
                    return False
                red.add(v)
            elif op == 'compute':
                if v in red:
                    return False
                if not all(p in red for p in parents[v]):
                    return False
                if len(red) >= budget:
                    return False
                red.add(v)
            elif op == 'store':
                if v not in red:
                    return False
                if v not in outputs:
                    return False
                red.remove(v)
                stored_outputs.add(v)
            elif op == 'delete':
                if v not in red:
                    return False
                if v in outputs:
                    return False
                red.remove(v)
            else:
                return False
        # All outputs must be stored
        if stored_outputs != set(outputs):
            return False
        return True

    # Try to parse dag structure
    nodes = None
    parents = None
    outputs = None

    try:
        if isinstance(dag, dict):
            if 'parents' in dag:
                parents = dag['parents']
            if 'outputs' in dag:
                outputs = dag['outputs']
            if isinstance(parents, list):
                nodes = list(range(len(parents)))
            elif isinstance(parents, dict):
                nodes = list(parents.keys())
        elif isinstance(dag, list):
            nodes = list(range(len(dag)))
            parents = []
            for i, node in enumerate(dag):
                if isinstance(node, dict):
                    p = node.get('parents', [])
                    parents.append(p)
                elif isinstance(node, (list, tuple)):
                    if len(node) >= 2 and isinstance(node[1], (list, tuple)):
                        parents.append(list(node[1]))
                    else:
                        parents.append([])
                else:
                    parents.append([])
            outputs = None
            for i, node in enumerate(dag):
                if isinstance(node, dict):
                    if node.get('is_output') or node.get('output'):
                        if outputs is None:
                            outputs = []
                        outputs.append(i)
    except Exception:
        pass

    if nodes is None or parents is None:
        return []

    # Normalize parents to list of lists of ints
    if isinstance(parents, dict):
        parent_list = []
        for i in range(len(nodes)):
            if i in parents:
                parent_list.append(list(parents[i]))
            else:
                parent_list.append([])
        parents = parent_list

    if outputs is None:
        all_parents = set()
        for p_list in parents:
            for p in p_list:
                all_parents.add(p)
        outputs = [i for i in range(len(nodes)) if i not in all_parents]
        if not outputs:
            outputs = [len(nodes) - 1]

    if not isinstance(outputs, list):
        outputs = [outputs]

    n = len(nodes)
    budget = fast_memory_budget

    # Build adjacency
    children = [[] for _ in range(n)]
    for i in range(n):
        for p in parents[i]:
            children[p].append(i)

    input_nodes = [i for i in range(n) if len(parents[i]) == 0]
    output_set = set(outputs)

    # Compute depth
    depth = [-1] * n
    in_degree = [len(parents[i]) for i in range(n)]
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        if depth[u] == -1:
            depth[u] = 0
        for v in children[u]:
            in_degree[v] -= 1
            depth[v] = max(depth[v], depth[u] + 1)
            if in_degree[v] == 0:
                queue.append(v)

    # For small DAGs, use bitmask DP over subsets of vertices
    # State: (red_mask, stored_mask) where red_mask is bitmask of red vertices
    # Cost: number of transfers (loads + stores)
    if n <= 16 and time_left() > 10.0:
        full_mask = (1 << n) - 1
        target_stored = 0
        for v in outputs:
            target_stored |= (1 << v)

        # Precompute valid operations for each vertex
        is_input = [1 if len(parents[v]) == 0 else 0 for v in range(n)]
        is_output = [1 if v in output_set else 0 for v in range(n)]
        parent_mask = [0] * n
        for v in range(n):
            for p in parents[v]:
                parent_mask[v] |= (1 << p)

        # Dijkstra over (red_mask, stored_mask)
        # State space: 2^n * 2^|outputs|
        start = (0, 0)
        target = (0, target_stored)

        # dist[state] = (cost, num_ops)
        dist = {start: (0, 0)}
        parent_map = {start: (None, None)}
        pq = [(0, 0, start)]

        max_states = 200000
        states_explored = 0

        while pq and states_explored < max_states and time_left() > 3.0:
            cost, num_ops, state = heapq.heappop(pq)

            if (cost, num_ops) > dist.get(state, (float('inf'), float('inf'))):
                continue

            states_explored += 1

            if state == target:
                # Reconstruct path
                path = []
                cur = state
                while parent_map[cur][0] is not None:
                    prev, op = parent_map[cur]
                    path.append(op)
                    cur = prev
                path.reverse()
                if len(path) <= 20000 and validate_schedule(path, parents, outputs, budget, n):
                    return path
                continue

            red_mask, stored_mask = state
            red_count = bin(red_mask).count('1')

            for v in range(n):
                v_bit = 1 << v
                is_red = (red_mask & v_bit) != 0
                v_input = is_input[v]
                v_output = is_output[v]
                parents_ok = (red_mask & parent_mask[v]) == parent_mask[v]

                # load(v): v must be blue, input, memory not full
                if not is_red and v_input and red_count < budget:
                    new_state = (red_mask | v_bit, stored_mask)
                    new_cost = cost + 1
                    new_ops = num_ops + 1
                    if new_ops <= 20000:
                        if new_state not in dist or (new_cost, new_ops) < dist[new_state]:
                            dist[new_state] = (new_cost, new_ops)
                            parent_map[new_state] = (state, ['load', v])
                            heapq.heappush(pq, (new_cost, new_ops, new_state))

                # compute(v): all parents red, v blue, memory not full
                if not is_red and parents_ok and red_count < budget:
                    new_state = (red_mask | v_bit, stored_mask)
                    new_cost = cost
                    new_ops = num_ops + 1
                    if new_ops <= 20000:
                        if new_state not in dist or (new_cost, new_ops) < dist[new_state]:
                            dist[new_state] = (new_cost, new_ops)
                            parent_map[new_state] = (state, ['compute', v])
                            heapq.heappush(pq, (new_cost, new_ops, new_state))

                # store(v): v red, output, not stored
                if is_red and v_output and not (stored_mask & v_bit):
                    new_state = (red_mask & ~v_bit, stored_mask | v_bit)
                    new_cost = cost + 1
                    new_ops = num_ops + 1
                    if new_ops <= 20000:
                        if new_state not in dist or (new_cost, new_ops) < dist[new_state]:
                            dist[new_state] = (new_cost, new_ops)
                            parent_map[new_state] = (state, ['store', v])
                            heapq.heappush(pq, (new_cost, new_ops, new_state))

                # delete(v): v red, not output
                if is_red and not v_output:
                    new_state = (red_mask & ~v_bit, stored_mask)
                    new_cost = cost
                    new_ops = num_ops + 1
                    if new_ops <= 20000:
                        if new_state not in dist or (new_cost, new_ops) < dist[new_state]:
                            dist[new_state] = (new_cost, new_ops)
                            parent_map[new_state] = (state, ['delete', v])
                            heapq.heappush(pq, (new_cost, new_ops, new_state))

    # Layered approach for larger DAGs
    max_depth = max(depth) if depth else 0
    layers = [[] for _ in range(max_depth + 1)]
    for v in range(n):
        if depth[v] >= 0:
            layers[depth[v]].append(v)

    # Compute max descendant depth
    max_desc_depth = [depth[i] for i in range(n)]
    for v in reversed(topo):
        for child in children[v]:
            max_desc_depth[v] = max(max_desc_depth[v], max_desc_depth[child])

    ops = []
    current_memory = set()

    # Load input nodes
    for v in input_nodes:
        if len(current_memory) < budget:
            ops.append(['load', v])
            current_memory.add(v)

    # Process layers
    for layer_idx in range(1, max_depth + 1):
        layer_nodes = layers[layer_idx]

        for v in layer_nodes:
            missing_parents = [p for p in parents[v] if p not in current_memory]
            if missing_parents:
                # Evict nodes no longer needed
                for u in list(current_memory):
                    if u not in output_set and max_desc_depth[u] < layer_idx:
                        ops.append(['delete', u])
                        current_memory.remove(u)
                        if not missing_parents:
                            break

                for p in missing_parents:
                    if len(current_memory) < budget:
                        ops.append(['load', p])
                        current_memory.add(p)

            if all(p in current_memory for p in parents[v]):
                if len(current_memory) < budget:
                    ops.append(['compute', v])
                    current_memory.add(v)

        # Delete nodes from previous layers no longer needed
        for u in list(current_memory):
            if depth[u] < layer_idx and u not in output_set:
                if max_desc_depth[u] <= layer_idx:
                    ops.append(['delete', u])
                    current_memory.remove(u)

    # Store outputs
    for v in outputs:
        if v in current_memory:
            ops.append(['store', v])
            current_memory.remove(v)

    # Delete remaining
    for v in list(current_memory):
        if v not in output_set:
            ops.append(['delete', v])
            current_memory.remove(v)

    if validate_schedule(ops, parents, outputs, budget, n):
        return ops

    # Fallback: naive topological order
    ops = []
    current_memory = set()

    for v in topo:
        if len(parents[v]) == 0:
            if len(current_memory) < budget:
                ops.append(['load', v])
                current_memory.add(v)
            else:
                for u in list(current_memory):
                    if u not in output_set:
                        ops.append(['delete', u])
                        current_memory.remove(u)
                        if len(current_memory) < budget:
                            break
                if len(current_memory) < budget:
                    ops.append(['load', v])
                    current_memory.add(v)
        else:
            missing = [p for p in parents[v] if p not in current_memory]
            if missing:
                for p in missing:
                    if len(current_memory) < budget:
                        ops.append(['load', p])
                        current_memory.add(p)
                    else:
                        for u in list(current_memory):
                            if u not in output_set and u not in parents[v]:
                                ops.append(['delete', u])
                                current_memory.remove(u)
                                if len(current_memory) < budget:
                                    break
                        if len(current_memory) < budget:
                            ops.append(['load', p])
                            current_memory.add(p)

            if all(p in current_memory for p in parents[v]):
                if len(current_memory) < budget:
                    ops.append(['compute', v])
                    current_memory.add(v)

    for v in outputs:
        if v in current_memory:
            ops.append(['store', v])
            current_memory.remove(v)

    for v in list(current_memory):
        if v not in output_set:
            ops.append(['delete', v])
            current_memory.remove(v)

    if validate_schedule(ops, parents, outputs, budget, n):
        return ops

    return []

# EVOLVE-BLOCK-END
