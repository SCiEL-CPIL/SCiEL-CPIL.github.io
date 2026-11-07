            for root_host in range(20):
                mapping = [-1] * k
                mapping[root] = root_host
                visited_host = {root_host}
                
                if dfs_tree(root, -1, mapping, visited_host, 1):
                    # ...
