        local_host_map = [-1] * k
        # ...
        if dfs_backtrack(0):
            # dfs_backtrack modified the `host_map` argument, NOT local_host_map
            # So local_host_map is still [-1]*k
            for u, v in tree:
                hu = local_host_map[u] # -1
                # ...
            return local_host_map # Returns [-1]*k
