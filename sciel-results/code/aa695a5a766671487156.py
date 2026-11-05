        other_edges = [e for e in cycle if e != edge_idx]
        if all(colors[e] != -1 for e in other_edges):
            c1, c2, c3 = (colors[e] for e in other_edges)
            if len({c1, c2, c3, new_color}) < 4:
                return False
