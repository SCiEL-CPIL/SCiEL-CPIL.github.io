        edge_list = list(initial_edges)
        # ...
        u, v = random.choice(edge_list)
        # ...
        # On swap accept:
        edge_list.remove((u, v)) # O(m)
        edge_list.append((x, y))
        # On revert:
        edge_list.remove((x, y))
        edge_list.append((u, v))
