<<<<<<< SEARCH
    def get_rank(self, n, m, p):
        if (n, m, p) in self.cache:
            return self.cache[(n, m, p)]
        
        best_rank = n * m * p
=======
    def get_rank(self, n, m, p):
        if (n, m, p) in self.cache:
            return self.cache[(n, m, p)]
        
        if n == 2 and m == 2 and p == 2:
            self.cache[(n, m, p)] = 7
            return 7
            
        best_rank = n * m * p
>>>>>>> REPLACE
