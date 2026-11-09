    N = 17
    
    def construct():
        def batcher_merge(wires):
            n = len(wires)
            if n <= 1:
                return []
            if n == 2:
                return [(wires[0], wires[1])]
            
            n1 = (n + 1) // 2
            n2 = n // 2
            
            wires1 = wires[:n1]
            wires2 = wires[n1:]
            
            network = []
            network.extend(batcher_merge(wires1))
            network.extend(batcher_merge(wires2))
            
            even_wires = [wires[i] for i in range(0, n, 2)]
            odd_wires = [wires[i] for i in range(1, n, 2)]
            
            network.extend(batcher_merge(even_wires))
            network.extend(batcher_merge(odd_wires))
            
            for i in range(0, n - 1, 2):
                network.append((wires[i], wires[i + 1]))
            
            return network
        
        wires = list(range(N))
        return batcher_merge(wires)
    