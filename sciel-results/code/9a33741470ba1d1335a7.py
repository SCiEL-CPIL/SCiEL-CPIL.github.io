def _edges_to_newick(edges, n_taxa, labels):
    if n_taxa == 1:
        return labels[0] + ";"
    if n_taxa == 2:
        return f"({labels[0]},{labels[1]});"
        
    adj = {i: [] for i in range(n_taxa)}
    max_node = max(max(u, v) for u, v in edges) if edges else n_taxa - 1
    for i in range(n_taxa, max_node + 1):
        adj[i] = []
        
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Root at leaf 0
    root = 0
    
    def dfs(node, parent, is_root=False):
        children = [c for c in adj[node] if c != parent]
        if not children:
            return labels[node]
        
        child_strs = [dfs(c, node, False) for c in children]
        
        if is_root and len(children) == 1:
            # Root is a leaf with one child
            # Return (child)root_label
            return f"({child_strs[0]}){labels[node]}"
        else:
            return "(" + ",".join(child_strs) + ")"
            
    newick_str = dfs(root, -1, True)
    return newick_str + ";"
