    def count_i6s(n, adj):
        all_mask = (1 << n) - 1
        total = 0
        for u in range(n):
            non_n = all_mask & ~(adj[u] | (1 << u))
            total += count_indep_4(non_n, adj)
        return total // 6
