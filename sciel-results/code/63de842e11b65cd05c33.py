"""New program using Gradient Boosted Decision Trees (GBDT) for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np

class SimpleDecisionTree:
    """
    A simple CART decision tree for regression.
    Optimized for 1D or low-dimensional inputs.
    """
    def __init__(self, max_depth=4, min_samples_split=5, min_samples_leaf=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree_ = None

    def fit(self, X, y):
        self.tree_ = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        
        # Check stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return {'leaf': True, 'value': np.mean(y)}
        
        # Check if all values are the same
        if np.var(y) < 1e-10:
            return {'leaf': True, 'value': np.mean(y)}
            
        best_gain = -1
        best_feature = -1
        best_threshold = None
        
        # Iterate over features
        for f in range(n_features):
            x_f = X[:, f]
            # Get unique values to determine potential split points
            # Using midpoints between sorted unique values is a standard robust approach
            unique_vals = np.unique(x_f)
            if len(unique_vals) < 2:
                continue
            
            # Consider a subset of thresholds if data is large, but here n=200 is small
            # We can check all midpoints
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
            
            for thresh in thresholds:
                left_mask = x_f <= thresh
                right_mask = ~left_mask
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                # Calculate SSE for this split
                sse_left = np.sum((y_left - np.mean(y_left))**2)
                sse_right = np.sum((y_right - np.mean(y_right))**2)
                sse_split = sse_left + sse_right
                
                # Calculate SSE for no split (parent)
                sse_parent = np.sum((y - np.mean(y))**2)
                
                gain = sse_parent - sse_split
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = f
                    best_threshold = thresh
        
        # If no valid split found, make a leaf
        if best_feature == -1:
            return {'leaf': True, 'value': np.mean(y)}
            
        # Create split
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }

    def predict_single(self, x, node):
        if node['leaf']:
            return node['value']
        
        if x[node['feature']] <= node['threshold']:
            return self.predict_single(x, node['left'])
        else:
            return self.predict_single(x, node['right'])

    def predict(self, X):
        if self.tree_ is None:
            return np.zeros(X.shape[0])
        
        predictions = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            predictions[i] = self.predict_single(X[i], self.tree_)
        
        return predictions

def fit(X_train, y_train):
    """
    Fits a Gradient Boosted Decision Tree model.
    """
    X = X_train
    y = y_train
    n_samples = X.shape[0]
    
    # GBDT Parameters
    n_estimators = 150
    learning_rate = 0.1
    max_depth = 4
    min_samples_split = 5
    min_samples_leaf = 2
    
    # Initialize predictions with the mean
    current_predictions = np.full(n_samples, np.mean(y))
    
    trees = []
    
    for i in range(n_estimators):
        # Residuals (negative gradient of MSE)
        residuals = y - current_predictions
        
        # Fit a tree to the residuals
        tree = SimpleDecisionTree(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
        tree.fit(X, residuals)
        
        # Predict residuals with the new tree
        residual_predictions = tree.predict(X)
        
        # Update predictions
        current_predictions += learning_rate * residual_predictions
        
        trees.append(tree)
        
        # Optional: Early stopping if improvement is minimal
        # Calculate current SSE
        sse = np.sum((y - current_predictions)**2)
        # We could store previous SSE and break if improvement < threshold
        # But fixed iterations is safer for determinism and speed control
        
    def predict(X_test):
        # Initialize with mean of training y
        preds = np.full(X_test.shape[0], np.mean(y))
        
        for tree in trees:
            preds += learning_rate * tree.predict(X_test)
            
        return preds

    return predict

# EVOLVE-BLOCK-END
