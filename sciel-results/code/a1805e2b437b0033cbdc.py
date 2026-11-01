    def get_valid_cluster(n):
        sp = math.sqrt(3)
        config = [[0.0, 0.0, 0.0]]
        # 6 neighbors
        for i in range(6):
            theta = math.radians(90 + 60 * i)
            config.append([sp * math.cos(theta), sp * math.sin(theta), 0.0])
        # 4 more in ring 2
        # Directions: 90, 270, 30, 210
        dirs = [90, 270, 30, 210]
        for d in dirs:
            theta = math.radians(d)
            config.append([2 * sp * math.cos(theta), 2 * sp * math.sin(theta), 0.0])
        
        # If n > 11, add more randomly far away (though N is fixed at 11 in this task)
        while len(config) < n:
            config.append([random.uniform(-5, 5), random.uniform(-5, 5), 0.0])
            
        return config[:n]
    