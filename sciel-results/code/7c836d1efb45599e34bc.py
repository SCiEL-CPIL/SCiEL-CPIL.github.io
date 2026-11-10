def add_edge(u, v, adj, edge_set):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)
    if u > v: u, v = v, u
    edge_set.add((u, v))

def remove_edge(u, v, adj, edge_set):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)
    if u > v: u, v = v, u
    edge_set.remove((u, v))
