    def find_q3(adj):
        for verts in all_8_subsets:
            edges = find_q3_edges(verts, adj)
            if edges:
                return edges
        return None
