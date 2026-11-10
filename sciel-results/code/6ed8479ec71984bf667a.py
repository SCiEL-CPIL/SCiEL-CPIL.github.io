    def greedy_init():
        nodes = sorted(valid_nodes_list, key=lambda x: adj[x].bit_count())
        current_set = set()
        blocked_mask = 0
        for node in nodes:
            if not (blocked_mask & (1 << node)):
                current_set.add(node)
                blocked_mask |= adj[node]
                blocked_mask |= (1 << node)
        return len(current_set), current_set
