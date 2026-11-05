    def add_edge(self, a, b):
        self.adj[a].append(b)
        self.adj[b].append(a)
        
    def remove_edge(self, a, b):
        if b in self.adj[a]:
            self.adj[a].remove(b)
        if a in self.adj[b]:
            self.adj[b].remove(a)
